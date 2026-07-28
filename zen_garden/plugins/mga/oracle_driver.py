"""ORACLE-mode driver for the MGA plugin.

Wires the projection machinery of plugin.MGA into pyoNearOpt's ORACLE
refinement loop (Turan, Moret, Bardow 2026). The paper's five steps map onto
this module as follows:

    Step 1 (initialise)  bounds and extreme designs from solve_axis_bounds,
                         turned into the initial outer box and the initial
                         inner point set here
    Step 2 (trial point) the max-min problem over the two approximations,
                         solved inside pyoNearOpt by the reformulation chosen
                         in oracle.max_min
    Step 3 (project)     MGA.find_nearest_point, one full model solve
    Step 4 (refine)      pyoNearOpt grows the inner hull and adds the cut
    Step 5 (terminate)   the artifacts written below, optionally after one
                         long certificate solve

The max-min problem supports two single-level reformulations, chosen by
oracle.max_min.formulation: "kkt_milp" is the published MILP (complementarity
via SOS1 or big-M), whose reported dual bound can stall at the t_max cap once
many inner points accumulate; "dual_bilinear" replaces the KKT system by LP
duality (no binaries, solved globally by Gurobi's nonconvex mode) with
identical witness and metric semantics but a bound that closes. After a
non-converged loop, one long max-min solve on the final geometry
(max_min.certificate_time_limit) can tighten the stored metric.

The solver is hardcoded to Gurobi: the max-min models need SOS1 constraints
(kkt_milp without big-M) or global nonconvex QP solving (dual_bilinear), and
the certificate reads Gurobi's dual bound directly.

pyoNearOpt compatibility: the base (published) package provides only the
default "kkt_milp" formulation; "dual_bilinear" needs a pyoNearOpt that
includes this formulation and fails otherwise. use_bigM defaults to true
because the base package is only sound with the big-M encoding: under SOS1
it reports the max-min INCUMBENT as the metric, which is not a valid upper
bound on the max-min distance. Only set use_bigM = false (SOS1) if your
pyoNearOpt reads the true dual bound (the version with the dual_bilinear
addition does).
"""

import importlib.metadata
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .polytope_io import Polytope, save_polytope

FORMULATIONS = ("kkt_milp", "dual_bilinear")


def run_oracle_mode(mga, oracle_cfg):
    """Run the ORACLE pipeline on a set-up MGA instance.

    Returns the summary directory holding the polytope npz and diagnostics.
    """
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm
    from pyoNearOpt.polytope_approximation.approximation_class import approximation

    if not mga.design_axes:
        raise ValueError(
            "MGA oracle: no exploration axes configured; set "
            "plugins.mga.axes.technologies and/or axes.carrier_imports."
        )
    tolerance = float(oracle_cfg["tolerance"])  # required, deliberately no default
    max_iterations = int(oracle_cfg.get("max_iterations", 200))
    initial_bounds = oracle_cfg.get("initial_bounds", "vmm")
    max_min_cfg = oracle_cfg.get("max_min", {})
    formulation = str(max_min_cfg.get("formulation", "kkt_milp")).lower()
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"MGA oracle: unknown max-min formulation {formulation!r}; "
            f"expected one of {list(FORMULATIONS)}."
        )
    # Only meaningful for kkt_milp; true is the sound default there (see
    # module docstring).
    use_bigM = bool(max_min_cfg.get("use_bigM", formulation == "kkt_milp"))

    # Step 1: bounds (and, with VMM, the extreme designs) must exist before
    # the projection model is added, because they define the coordinates.
    supplied = None if initial_bounds == "vmm" else initial_bounds
    mga.solve_axis_bounds(supplied)
    initial_points, point_origin = mga.initial_inner_points()
    mga.setup_projection_model(initial_points=initial_points)

    with _coordinate_warnings_suppressed():
        A0, b0 = mga.build_initial_outer_approximation()
        approximation_kwargs = dict(
            A=A0,
            X=initial_points,
            b=b0,
            name_list=list(mga.z_names),
            use_bigM=use_bigM,
        )
        # Only a pyoNearOpt with the dual_bilinear addition knows the
        # keyword; the base package's sole formulation is kkt_milp, so
        # omitting it is equivalent.
        if formulation != "kkt_milp":
            approximation_kwargs["formulation"] = formulation
        poly = approximation(**approximation_kwargs)
        # big_M bounds the KKT duals (pyoNearOpt default 1e3, used only with
        # use_bigM). It must exceed the true duals or optima are cut off;
        # pyoNearOpt raises when a dual reaches it, so the default here is
        # simply generous. t_max caps the max-min distance (default 1e6).
        poly.Md = float(max_min_cfg.get("big_M", 1e8))
        if max_min_cfg.get("t_max") is not None:
            poly.t_max = float(max_min_cfg["t_max"])

    # Gurobi options for the max-min solves: solver defaults unless configured.
    solver_options = dict(max_min_cfg.get("solver_options", {}))
    if formulation == "dual_bilinear":
        # The bilinear objective needs Gurobi's global nonconvex-QP mode.
        solver_options.setdefault("NonConvex", 2)
    # Gurobi output otherwise reaches the run log twice: once as console text
    # and once through the 'gurobipy' logger. Keep only the console copy.
    logging.getLogger("gurobipy").propagate = False

    algo = ORACLEAlgorithm(
        poly_approx=poly,
        find_nearest_point=mga.find_nearest_point,
        max_iterations=max_iterations,
        tol=tolerance,
        pyomo_solver=_gurobi_solver(solver_options),
        print_lv=1,
    )
    logging.info(
        f"MGA oracle: max-min formulation = {formulation!r}, tol = "
        f"{tolerance:.3g}, initial bounds = "
        f"{'VMM' if supplied is None else 'supplied'}"
    )

    # The summary folder is a sibling of the per-iteration Postprocess
    # folders; the subfolder separates scenarios (empty for plain runs).
    out = (
        Path(mga.optimization_setup.analysis.folder_output)
        / f"{mga.postprocess_ctx['model_name']}_oracle_summary"
        / mga.postprocess_ctx["subfolder"]
    )
    out.mkdir(parents=True, exist_ok=True)
    df = None
    metric_source = "loop"
    iterations_done = 0
    try:
        with _coordinate_warnings_suppressed():
            df = algo.refine_approximations()
        iterations_done = 0 if df is None else int(len(df))
        df, metric_source = _run_final_certificate(
            df, poly, tolerance, max_min_cfg, solver_options
        )
    finally:
        # Persist artifacts even if the loop raised mid-way (df stays None).
        run_info = {
            "formulation": formulation,
            "use_bigM": use_bigM,
            "max_iterations": max_iterations,
            "iterations_done": iterations_done,
            "metric_source": metric_source,
            "initial_bounds": "vmm" if supplied is None else "supplied",
            "versions": _package_versions(),
        }
        _save_artifacts(mga, poly, df, tolerance, out, point_origin, run_info)
    return out


def _run_final_certificate(df, poly, tolerance, max_min_cfg, solver_options):
    """One long max-min solve on the final geometry after a non-converged loop.

    The per-iteration max-min solves get a short time limit because the loop
    only needs the next trial point; the proof (the reported metric) is
    cheaper bought once, at the end. The loop's last reported metric is a
    valid cap for this solve (with the base pyoNearOpt this requires
    use_bigM, see the module docstring); the certified value is appended as
    an extra diagnostics row so the saved npz carries the best-known metric.
    Controlled by oracle.max_min.certificate_time_limit (seconds, 0 = off).

    Returns (diagnostics, metric_source).
    """
    time_limit = float(max_min_cfg.get("certificate_time_limit", 0) or 0)
    if df is None or len(df) == 0 or time_limit <= 0:
        return df, "loop"
    last = float(df["max_min_distance"].iloc[-1])
    if last <= tolerance:
        return df, "loop"

    logging.info(
        f"MGA oracle: final certificate solve "
        f"(TimeLimit = {time_limit:.0f}s, cap = {last:.4g})."
    )
    try:
        poly.t_max = last
        poly.inner_outer_model()
        solver = _gurobi_solver(dict(solver_options, TimeLimit=time_limit))
        # load_solutions=False: only the dual bound is needed, and a
        # solution-less timeout must not raise.
        solver.solve(poly.out_inner, load_solutions=False, tee=True)
        certified = min(last, -float(solver._solver_model.ObjBound))
    except Exception:
        logging.exception(
            "MGA oracle: final certificate failed; keeping the loop's metric."
        )
        return df, "loop"

    logging.info(
        f"MGA oracle: final certificate {last:.4g} -> {certified:.4g} "
        f"(tol = {tolerance:.4g})."
    )
    row = {c: None for c in df.columns}
    row["max_min_distance"] = certified
    if "iteration" in df.columns:
        row["iteration"] = int(df["iteration"].iloc[-1]) + 1
    if "max_min_solve_time" in df.columns:
        row["max_min_solve_time"] = time_limit
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True), "certificate"


def _save_artifacts(mga, poly, df, tolerance, out, point_origin, run_info):
    """Persist the polytope npz + diagnostics csv and log the outcome.

    Runs in run_oracle_mode's `finally`, so the completed iterations survive
    a mid-loop exception (df is None in that case).
    """
    if df is not None:
        final_distance = float(df["max_min_distance"].iloc[-1])
        converged = bool(final_distance <= tolerance)
    else:
        final_distance = float("nan")
        converged = False

    meta = mga.polytope_metadata()
    # X grows by one certified point per iteration; the rows beyond the
    # initial set are those iterates.
    origins = list(point_origin)
    origins += ["iterate"] * (poly.X.shape[0] - len(origins))
    save_polytope(
        out / "polytope.npz",
        Polytope(
            A=poly.A,
            b=poly.b,
            X=poly.X,
            names=list(mga.z_names),
            kinds=[axis.kind for axis in mga.axes],
            units=[axis_meta["unit"] or "" for axis_meta in meta["axes"]],
            scale=mga.scale,
            offset=mga.offset,
            bounds_phys=mga.bounds_phys,
            z_star_phys=mga.z_star_phys,
            c_star=float(mga.c_star),
            epsilon=float(mga.epsilon),
            tolerance=float(tolerance),
            converged=converged,
            final_max_min_distance=float(final_distance),
            n_initial_rows=int(mga.n_initial_rows),
            point_origin=origins,
            meta=meta,
            run=run_info,
        ),
    )
    if df is not None:
        df.to_csv(out / "diagnostics.csv", index=False)
        log = logging.info if converged else logging.warning
        log(
            f"MGA oracle: {'CONVERGED' if converged else 'did NOT converge'} "
            f"after {len(df)} iterations, final max_min_distance = "
            f"{final_distance:.4g} (tol = {tolerance:.4g})."
        )
    else:
        logging.warning(
            "MGA oracle: no diagnostics to save (the refinement "
            "loop raised); see traceback above."
        )
    logging.info(f"MGA oracle: artifacts saved to {out}")


def _package_versions() -> dict:
    """Versions of the packages that determine a run's results."""
    versions = {}
    for package in ("zen-garden", "pyoNearOpt", "linopy", "gurobipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _gurobi_solver(options):
    """Pyomo Gurobi solver on the Python API (no LP file round-trips).

    manage_env=True creates and releases the Gurobi environment together with
    the solver object, so single-use academic licenses are not held open.
    """
    import pyomo.environ as pyo

    solver = pyo.SolverFactory("gurobi", solver_io="python", manage_env=True)
    solver.set_options(" ".join(f"{k}={v}" for k, v in options.items()))
    return solver


@contextmanager
def _coordinate_warnings_suppressed():
    """Silence linopy's coordinate-mismatch UserWarning: the MGA projection
    variables intentionally live on different coordinates than the model's."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Coordinates across variables not equal.*",
            category=UserWarning,
        )
        yield
