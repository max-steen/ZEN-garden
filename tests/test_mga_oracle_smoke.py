"""End-to-end smoke test for the MGA oracle plugin.

Runs the two Crystal_Ball_small smoke configs through ``python -m zen_garden``
and asserts PROPERTIES of the produced polytope npz (key set, shapes,
containment, metadata consistency) — not byte identity, so it stays green
across refactors that only change degenerate-optimum tie-breaking.

Each config runs in its own subprocess: the mga plugin keeps module-level
config state that shallow-merges and would leak keys across two in-process
runner.run() calls.

Gated: set RUN_MGA_SMOKE=1 to run (each config takes ~10 min of Gurobi time).
Set MGA_SMOKE_OUT=<dir> to keep the outputs (e.g. for baseline comparisons)
instead of writing to pytest's tmp_path.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(os.environ.get(
    "MGA_SMOKE_DATA", "/Users/maxsteen/zen-work/datasets/ZEN-models-Crystal_Ball/data"
))
DATASET = DATA_DIR / "Crystal_Ball_small"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_MGA_SMOKE") != "1",
        reason="set RUN_MGA_SMOKE=1 to run (slow, needs Gurobi)",
    ),
    pytest.mark.skipif(not DATASET.is_dir(), reason=f"dataset missing: {DATASET}"),
]

# Hardcoded on purpose (not imported from polytope_io): the test must fail if
# the writer's schema drifts from this contract.
EXPECTED_KEYS = {
    "A", "b", "X", "name_list", "u_star", "c_star", "epsilon", "cost_axis",
    "z_star", "units", "tolerance", "converged", "final_max_min_distance",
    "axis_meta_json",
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
    npz_path = next(summary.glob("polytope*.npz"))  # run-id suffix varies
    return summary, npz_path


def _assert_polytope_properties(summary: Path, npz_path: Path, mga_cfg: dict):
    """Common property assertions; returns (names, meta, cost_axis)."""
    d = np.load(npz_path)  # schema has no object arrays; allow_pickle stays False
    assert set(d.files) == EXPECTED_KEYS, f"npz keys: {sorted(d.files)}"

    A = d["A"]
    b = np.asarray(d["b"], dtype=float).ravel()
    X = d["X"]
    u_star = np.asarray(d["u_star"], dtype=float)
    z_star = np.asarray(d["z_star"], dtype=float)
    names = [str(n) for n in d["name_list"]]
    units = [str(u) for u in d["units"]]
    cost_axis = str(d["cost_axis"])

    n_z = u_star.shape[0]
    n_explore = n_z + (1 if cost_axis else 0)

    # Shape consistency.
    assert A.ndim == 2 and A.shape[1] == n_explore
    assert b.shape == (A.shape[0],)
    assert X.ndim == 2 and X.shape[1] == n_explore and X.shape[0] >= 1
    assert z_star.shape == (n_z,)
    assert len(names) == n_explore == len(units)

    # Every stored feasible point satisfies the outer approximation.
    tol = 1e-6 * (1.0 + np.abs(b).max())
    violations = X @ A.T - b
    assert (violations <= tol).all(), (
        f"containment violated: max violation {violations.max():.3e} > tol {tol:.3e}"
    )

    # Scalars and normalisation preconditions.
    assert np.isfinite(u_star).all() and (u_star > 0).all()
    assert (z_star >= 0).all()
    assert d["converged"].dtype == np.bool_
    assert float(d["epsilon"]) == float(mga_cfg["epsilon"])
    assert float(d["tolerance"]) == float(mga_cfg["oracle"]["tolerance"])

    # Axis metadata agrees with name_list.
    meta = json.loads(str(d["axis_meta_json"]))
    assert [a["name"] for a in meta["axes"]] == names[:n_z]
    assert all(a["kind"] in ("tech_capacity", "carrier_import") for a in meta["axes"])
    assert meta["include_cost"] == bool(cost_axis)
    if cost_axis:
        assert names[-1] == cost_axis == meta["cost_axis"]
    for a in meta["axes"]:
        assert isinstance(a["members"], list) and a["members"]
        assert (a["capacity_type"] is None) == (a["kind"] == "carrier_import")

    # Per-iteration diagnostics were written (header + at least one data row).
    diag = summary / "diagnostics.csv"
    assert diag.is_file()
    assert len(diag.read_text().strip().splitlines()) >= 2

    return names, meta, cost_axis


def test_oracle_smoke_full(tmp_path):
    """Lumped techs + carrier-import axis + cost coordinate (production code path)."""
    cfg_path = DATA_DIR / "config_smoke_full.json"
    mga_cfg = json.loads(cfg_path.read_text())["plugins"]["mga"]
    summary, npz_path = _run_smoke(cfg_path, _out_dir(tmp_path, "full"))
    names, meta, cost_axis = _assert_polytope_properties(summary, npz_path, mga_cfg)

    assert names[:5] == ["nuclear", "photovoltaics", "battery", "hydro_lump",
                         "biomass"]
    assert [a["kind"] for a in meta["axes"]] == (["tech_capacity"] * 4
                                                 + ["carrier_import"])
    assert meta["axes"][3]["members"] == ["reservoir_hydro", "run-of-river_hydro"]
    assert cost_axis == "net_present_cost"


def test_oracle_smoke_legacy(tmp_path):
    """exclude_techs config: singleton tech axes, no cost coordinate."""
    cfg_path = DATA_DIR / "config_smoke_legacy.json"
    mga_cfg = json.loads(cfg_path.read_text())["plugins"]["mga"]
    summary, npz_path = _run_smoke(cfg_path, _out_dir(tmp_path, "legacy"))
    names, meta, cost_axis = _assert_polytope_properties(summary, npz_path, mga_cfg)

    assert cost_axis == ""
    assert all(a["kind"] == "tech_capacity" for a in meta["axes"])
    assert all(a["members"] == [a["name"]] for a in meta["axes"])
    assert len(names) == 8  # the 8 kept singleton axes
