"""ORACLE-mode driver for the MGA plugin.

Wires the projection machinery of plugin.MGA into pyoNearOpt's ORACLE
refinement loop (Turan, Moret, Bardow 2026): per-axis fmax LPs (normalisation
and initial outer box), then the L-infinity projection model, then the
iterative refinement of inner/outer polytope approximations, then artifacts
(polytope npz + diagnostics csv).

Step 2 of ORACLE (the max-min trial-point problem) supports two single-level
reformulations, chosen by oracle.formulation: "kkt_milp" is the published
MILP (complementarity via SOS1 or Big-M), whose reported dual bound can stall
at the t_max cap once many inner points accumulate; "dual_bilinear" replaces
the KKT system by LP duality (no binaries, solved globally by Gurobi's
nonconvex mode) with identical witness and metric semantics but a bound that
closes. After a non-converged loop, one long step-2 solve on the final
geometry (oracle.final_certificate_time_limit) can tighten the stored metric.
The solver is hardcoded to Gurobi.

pyoNearOpt compatibility: with the base (published) pyoNearOpt only the
default "kkt_milp" formulation exists ("dual_bilinear" needs the patched
package and fails otherwise), and the base package must be run with
oracle.use_bigM: true -- under the SOS1 encoding it reports the step-2
INCUMBENT as the metric, which is not a valid upper bound on the max-min
distance, so the convergence claim, the in-loop t_max cap, and the final
certificate's cap would all rest on an unsound value.
"""

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .polytope_io import Polytope, save_polytope


def run_oracle_mode(mga, oracle_cfg):
    """Run the ORACLE pipeline on a set-up MGA instance.

    Returns the summary directory holding the polytope npz and diagnostics.
    """
    # Imported lazily so weights mode works without pyoNearOpt installed.
    from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm
    from pyoNearOpt.polytope_approximation.approximation_class import approximation

    tolerance = float(oracle_cfg["tolerance"])  # required, deliberately no default
    max_iterations = oracle_cfg.get("max_iterations", 200)
    formulation = str(oracle_cfg.get("formulation", "kkt_milp")).lower()
    if formulation not in ("kkt_milp", "dual_bilinear"):
        raise ValueError(f"MGA oracle: unknown formulation {formulation!r}")
    use_bigM = bool(oracle_cfg.get("use_bigM", False))
    if formulation == "dual_bilinear" and use_bigM:
        logging.warning(
            "MGA oracle: use_bigM is ignored with formulation='dual_bilinear'."
        )
    vmm_init = bool(oracle_cfg.get("vmm_initialization", False))

    # The fmax LPs must run before the projection model is added: they provide
    # the normalisation denominators and the initial outer box. The optional
    # fmin LPs complete VMM, so that all 2*n_z extreme designs seed the
    # initial inner approximation.
    mga.compute_fmax_normalization()
    if vmm_init:
        mga.solve_extreme_lps("min")
    initial_points = mga.initial_inner_points(include_extreme_designs=vmm_init)
    mga.setup_projection_model(initial_points=initial_points)

    with _coordinate_warnings_suppressed():
        A0, b0 = mga.build_initial_outer_approximation()
        name_list = list(mga.z_names)
        if mga.include_cost:
            name_list.append("net_present_cost")
        approximation_kwargs = dict(
            A=A0, X=initial_points, b=b0,
            name_list=name_list,
            use_bigM=use_bigM,
        )
        # Only the patched pyoNearOpt knows the keyword; the base package's
        # sole formulation is kkt_milp, so omitting it is equivalent.
        if formulation != "kkt_milp":
            approximation_kwargs["formulation"] = formulation
        poly = approximation(**approximation_kwargs)
        # Md is pyoNearOpt's big-M on the KKT duals (default 1e3, used only
        # with use_bigM). It must upper-bound the true duals or optima are
        # silently cut off; pyoNearOpt raises when a dual hits it, so the
        # default here is simply generous. t_max caps the max-min distance
        # (pyoNearOpt default 1e6).
        poly.Md = float(oracle_cfg.get("Md_override", 1e8))
        if oracle_cfg.get("t_max_override") is not None:
            poly.t_max = float(oracle_cfg["t_max_override"])

    # Gurobi options for the step-2 solves: solver defaults unless set in
    # the config (production runs should set at least a TimeLimit).
    milp_options = dict(oracle_cfg.get("milp_options", {}))
    if formulation == "dual_bilinear":
        # The bilinear objective needs Gurobi's global nonconvex-QP mode.
        milp_options.setdefault("NonConvex", 2)
    # Gurobi output otherwise reaches the run log twice: once as console text
    # and once through the 'gurobipy' logger. Keep only the console copy.
    logging.getLogger("gurobipy").propagate = False

    algo = ORACLEAlgorithm(
        poly_approx=poly,
        find_nearest_point=mga.find_nearest_point,
        max_iterations=max_iterations,
        tol=tolerance,
        pyomo_solver=_gurobi_solver(milp_options),
        print_lv=1,
    )
    logging.info(f"MGA oracle: formulation = {formulation!r}, tol = {tolerance:.3g}")

    # The summary folder is a sibling of the per-iteration Postprocess folders.
    out = (Path(mga.optimization_setup.analysis.folder_output)
           / f"{mga.postprocess_ctx['model_name']}_oracle_summary")
    out.mkdir(parents=True, exist_ok=True)
    df = None
    try:
        with _coordinate_warnings_suppressed():
            df = algo.refine_approximations()
        df = _run_final_certificate(df, poly, tolerance, oracle_cfg, milp_options)
    finally:
        # Persist artifacts even if the loop raised mid-way (df stays None).
        _save_artifacts(mga, poly, df, tolerance, out)
    return out


def _run_final_certificate(df, poly, tolerance, oracle_cfg, milp_options):
    """One long step-2 solve on the final geometry after a non-converged loop.

    The per-iteration step-2 solves get a short TimeLimit because the loop
    only needs the next trial point; the proof (the reported metric) is
    cheaper bought once, at the end. The loop's last reported metric is a
    valid cap for this solve (with the base pyoNearOpt this requires
    use_bigM, see the module docstring); the certified value is appended as
    an extra diagnostics row so the saved npz carries the best-known metric.
    Controlled by oracle.final_certificate_time_limit (seconds, 0 = off).
    """
    time_limit = float(oracle_cfg.get("final_certificate_time_limit", 0) or 0)
    if df is None or len(df) == 0 or time_limit <= 0:
        return df
    last = float(df["max_min_distance"].iloc[-1])
    if last <= tolerance:
        return df

    logging.info(f"MGA oracle: final certificate solve "
                 f"(TimeLimit = {time_limit:.0f}s, cap = {last:.4g}).")
    try:
        poly.t_max = last
        poly.inner_outer_model()
        solver = _gurobi_solver(dict(milp_options, TimeLimit=time_limit))
        # load_solutions=False: only the dual bound is needed, and a
        # solution-less timeout must not raise.
        solver.solve(poly.out_inner, load_solutions=False, tee=True)
        certified = min(last, -float(solver._solver_model.ObjBound))
    except Exception:
        logging.exception(
            "MGA oracle: final certificate failed; keeping the loop's metric."
        )
        return df

    logging.info(f"MGA oracle: final certificate {last:.4g} -> {certified:.4g} "
                 f"(tol = {tolerance:.4g}).")
    row = {c: None for c in df.columns}
    row["max_min_distance"] = certified
    if "iteration" in df.columns:
        row["iteration"] = int(df["iteration"].iloc[-1]) + 1
    if "max_min_solve_time" in df.columns:
        row["max_min_solve_time"] = time_limit
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def _save_artifacts(mga, poly, df, tolerance, out):
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
    unit_by_name = {a["name"]: (a["unit"] or "") for a in meta["axes"]}
    if meta["cost_axis"]:
        unit_by_name[meta["cost_axis"]] = meta["cost_unit"] or ""
    # Name the file after the run (last "_"-token of the output folder, e.g.
    # ".../my_model_06" -> polytope_06.npz) so collected files stay
    # self-identifying.
    run_id = Path(mga.optimization_setup.analysis.folder_output).name.split("_")[-1]
    polytope_file = f"polytope_{run_id}.npz" if run_id else "polytope.npz"
    save_polytope(out / polytope_file, Polytope(
        A=poly.A, b=poly.b, X=poly.X,
        names=[str(n) for n in poly.name_list],
        u_star=mga.u_star,
        c_star=float(mga.c_star),
        epsilon=float(mga.epsilon),
        cost_axis=meta["cost_axis"] or "",
        z_star=mga.z_star_raw,
        units=[unit_by_name.get(str(n), "") for n in poly.name_list],
        tolerance=float(tolerance),
        converged=converged,
        final_max_min_distance=float(final_distance),
        meta=meta,
    ))
    if df is not None:
        df.to_csv(out / "diagnostics.csv", index=False)
        log = logging.info if converged else logging.warning
        log(f"MGA oracle: {'CONVERGED' if converged else 'did NOT converge'} "
            f"after {len(df)} iterations, final max_min_distance = "
            f"{final_distance:.4g} (tol = {tolerance:.4g}).")
    else:
        logging.warning("MGA oracle: no diagnostics to save (the refinement "
                        "loop raised); see traceback above.")
    logging.info(f"MGA oracle: artifacts saved to {out} (polytope: {polytope_file})")


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
