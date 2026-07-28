"""End-to-end smoke test for the MGA oracle plugin.

Runs the two Crystal_Ball_small smoke configs through ``python -m zen_garden``
and asserts PROPERTIES of the produced polytope npz (schema, shapes,
containment, normalisation, metadata consistency) — not byte identity, so it
stays green across refactors that only change degenerate-optimum tie-breaking.

Each config runs in its own subprocess: the mga plugin keeps module-level
config state that shallow-merges and would leak keys across two in-process
runner.run() calls.

Gated: set RUN_MGA_SMOKE=1 to run (each config needs several minutes of
Gurobi time) and MGA_SMOKE_DATA to the dataset directory. MGA_SMOKE_OUT
keeps the outputs instead of writing to pytest's tmp_path.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_DATA = os.environ.get("MGA_SMOKE_DATA")
DATA_DIR = Path(_DATA) if _DATA else None
DATASET = DATA_DIR / "Crystal_Ball_small" if DATA_DIR else None

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_MGA_SMOKE") != "1",
        reason="set RUN_MGA_SMOKE=1 to run (slow, needs Gurobi)",
    ),
    pytest.mark.skipif(
        DATASET is None or not DATASET.is_dir(),
        reason="set MGA_SMOKE_DATA to the directory holding Crystal_Ball_small "
        "and the config_smoke_*.json files",
    ),
]

# Hardcoded on purpose (not imported from polytope_io): the test must fail if
# the writer's schema drifts from this contract.
EXPECTED_KEYS = {
    "A", "b", "X", "name_list", "kinds", "units", "scale",
    "offset", "bounds_phys", "z_star_phys", "c_star", "epsilon", "tolerance",
    "converged", "final_max_min_distance", "n_initial_rows", "point_origin",
    "axis_meta_json", "run_json",
}


def _out_dir(tmp_path: Path, tag: str) -> Path:
    base = os.environ.get("MGA_SMOKE_OUT")
    if base:
        d = Path(base) / tag
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path


def _run_smoke(config_path: Path, out_dir: Path):
    """Run zen_garden on `config_path`; return (summary_dir, npz_path)."""
    res = subprocess.run(
        [sys.executable, "-m", "zen_garden",
         f"--config={config_path}", f"--folder_output={out_dir}"],
        # dataset paths resolve relative to the config file; the cwd only
        # keeps gurobi.log out of the repo
        cwd=out_dir,
        capture_output=True, text=True, timeout=3600,
    )
    assert res.returncode == 0, (
        f"zen_garden failed (rc={res.returncode}):\n"
        f"--- stdout tail ---\n{res.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{res.stderr[-4000:]}"
    )
    summary = next(out_dir.glob("*_oracle_summary"))
    return summary, summary / "polytope.npz"


def _assert_polytope_properties(summary: Path, npz_path: Path, mga_cfg: dict):
    """Common property assertions; returns (names, kinds, meta)."""
    d = np.load(npz_path)  # the schema has no object arrays; allow_pickle stays False
    assert set(d.files) == EXPECTED_KEYS, f"npz keys: {sorted(d.files)}"

    A = d["A"]
    b = np.asarray(d["b"], dtype=float).ravel()
    X = d["X"]
    scale = np.asarray(d["scale"], dtype=float)
    offset = np.asarray(d["offset"], dtype=float)
    bounds = np.asarray(d["bounds_phys"], dtype=float)
    z_star = np.asarray(d["z_star_phys"], dtype=float)
    names = [str(n) for n in d["name_list"]]
    kinds = [str(k) for k in d["kinds"]]
    units = [str(u) for u in d["units"]]
    origins = [str(o) for o in d["point_origin"]]
    n_initial = int(d["n_initial_rows"])
    n_axes = len(names)

    # Shape consistency across every per-axis array.
    assert A.ndim == 2 and A.shape[1] == n_axes
    assert b.shape == (A.shape[0],)
    assert X.ndim == 2 and X.shape[1] == n_axes and X.shape[0] >= 1
    assert len(kinds) == len(units) == n_axes
    assert scale.shape == offset.shape == z_star.shape == (n_axes,)
    assert bounds.shape == (n_axes, 2)
    assert len(origins) == X.shape[0]

    # Every stored feasible point satisfies the outer approximation.
    tol = 1e-6 * (1.0 + np.abs(b).max())
    violations = X @ A.T - b
    assert (violations <= tol).all(), (
        f"containment violated: max violation {violations.max():.3e} > tol {tol:.3e}"
    )

    # The initial rows are the box, unit-normalised: |row| == 1 and every
    # upper row reads z <= 1.
    assert n_initial == 2 * n_axes
    initial_norms = np.linalg.norm(A[:n_initial], axis=1)
    assert np.allclose(initial_norms, 1.0), initial_norms
    assert np.allclose(b[n_axes:n_initial], 1.0), b[n_axes:n_initial]
    assert (b[:n_axes] <= 1e-12).all()  # lower rows: -z <= -lower <= 0

    # Normalisation is consistent with the stored bounds and z*.
    assert (scale > 0).all()
    assert np.allclose((bounds[:, 1] - offset) / scale, 1.0)
    assert (z_star >= -1e-9).all()
    assert np.allclose(X[0], (z_star - offset) / scale, atol=1e-9)

    # Provenance: X starts at z*, then the VMM extreme designs, then iterates.
    assert origins[0] == "z_star"
    n_design = sum(1 for k in kinds if k != "total_cost")
    assert sum(o.startswith("max:") for o in origins) == n_design
    assert sum(o.startswith("min:") for o in origins) == n_design
    # everything past z* and the 2 * n_design extreme designs is a loop iterate
    assert set(origins[1 + 2 * n_design:]) <= {"iterate"}

    # Scalars and run provenance.
    assert d["converged"].dtype == np.bool_
    assert float(d["epsilon"]) == float(mga_cfg["epsilon"])
    assert float(d["tolerance"]) == float(mga_cfg["oracle"]["tolerance"])
    run = json.loads(str(d["run_json"]))
    assert run["initial_bounds"] == "vmm"
    assert run["formulation"] in ("kkt_milp", "dual_bilinear")
    assert run["metric_source"] in ("loop", "certificate")
    assert run["iterations_done"] >= 1

    # Axis metadata agrees with name_list and kinds.
    meta = json.loads(str(d["axis_meta_json"]))
    assert [a["name"] for a in meta["axes"]] == names
    assert [a["kind"] for a in meta["axes"]] == kinds
    for axis in meta["axes"]:
        assert axis["kind"] in ("tech_capacity", "carrier_import", "total_cost")
        # only technology axes carry a capacity type; only design axes members
        assert (axis["capacity_type"] is None) == (axis["kind"] != "tech_capacity")
        assert bool(axis["members"]) == (axis["kind"] != "total_cost")

    # Per-iteration diagnostics were written (header + at least one data row).
    diag = summary / "diagnostics.csv"
    assert diag.is_file()
    assert len(diag.read_text().strip().splitlines()) >= 2

    return names, kinds, meta


def test_oracle_smoke_full(tmp_path):
    """Lumped techs + carrier-import axis + cost axis (production code path)."""
    cfg_path = DATA_DIR / "config_smoke_full.json"
    mga_cfg = json.loads(cfg_path.read_text())["plugins"]["mga"]
    summary, npz_path = _run_smoke(cfg_path, _out_dir(tmp_path, "full"))
    names, kinds, meta = _assert_polytope_properties(summary, npz_path, mga_cfg)

    assert names == ["nuclear", "photovoltaics", "battery", "hydro_lump",
                     "biomass", "net_present_cost"]
    assert kinds == ["tech_capacity"] * 4 + ["carrier_import", "total_cost"]
    assert meta["axes"][3]["members"] == ["reservoir_hydro", "run-of-river_hydro"]


def test_oracle_smoke_basic(tmp_path):
    """Singleton technology axes only, no cost axis."""
    cfg_path = DATA_DIR / "config_smoke_basic.json"
    mga_cfg = json.loads(cfg_path.read_text())["plugins"]["mga"]
    summary, npz_path = _run_smoke(cfg_path, _out_dir(tmp_path, "basic"))
    names, kinds, meta = _assert_polytope_properties(summary, npz_path, mga_cfg)

    assert len(names) == 8
    assert set(kinds) == {"tech_capacity"}
    assert all(a["members"] == [a["name"]] for a in meta["axes"])
