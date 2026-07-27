"""
MGA (Modelling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` handler that adds a near-optimality cost budget
(cost <= (1 + epsilon) * C*) to the solved baseline model and re-solves it
under one of two modes:

* "weights": one solve per user-provided weight dict, each minimising
  sum_i w_i * capacity_addition_i.
* "oracle": the ORACLE algorithm (Turan, Moret, Bardow 2026) iteratively
  refines inner and outer polytope approximations of the near-optimal space
  via L-infinity projections. The refinement loop lives in pyoNearOpt; the
  projection solves run on the ZEN-garden model (driver in oracle.py).

Oracle-mode exploration axes are groups of technologies (summed capacity
addition) or carriers (duration-weighted annual import), see axes.py. All
polytope coordinates are normalised: design axis i is z_i / U_i*, with U_i*
from one fmax LP per axis, and the optional cost axis is
(C - C*) / (epsilon * C*). Every solve is written to disk as a sibling
sub-solution of the baseline via Postprocess.

Config (the "plugins.mga" block in config.json):
    epsilon (float): near-optimality slack, default 0.1.
    mode (str): "weights" (default) or "oracle".
    iterations (list[dict]): weights mode; one {"weights": {tech: w}} dict
        per iteration.
    include_techs (list): tech axes; entries are technology names or
        single-key dicts {group_name: [members]} for lumped axes. Mutually
        exclusive with exclude_techs (singleton axes for every technology
        not listed).
    include_carrier_imports (list): carrier-import axes, same entry format.
    oracle (dict): tolerance (required), max_iterations (default 200),
        formulation ("kkt_milp" | "dual_bilinear"), include_cost (bool),
        milp_options (Gurobi options for ORACLE's internal step-2 solves),
        Md_override, t_max_override, use_bigM,
        final_certificate_time_limit (seconds, 0 = off). See oracle.py.
"""

import logging
import time

import numpy as np
import xarray as xr

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess

from .axes import Axis, axis_physical_unit, build_axis_groups, cost_physical_unit
from .oracle import run_oracle_mode
from .polytope_io import CARRIER_IMPORT, TECH_CAPACITY

# Module-level config, updated by the plugin loader with the user's
# "plugins.mga" block. The merge is a SHALLOW dict.update: nested dicts like
# "oracle" are replaced wholesale, so their defaults must be applied at
# access time via .get(), never stored here.
config = {
    "epsilon": 0.1,
    "mode": "weights",
    "exclude_techs": [],
    "include_techs": [],
    "include_carrier_imports": [],
    "iterations": [],
    "oracle": {},
}

# A cut returned by find_nearest_point must keep every known near-optimal
# point inside the outer approximation; inexact projection duals (e.g.
# barrier without crossover) can violate that. Violations above this trigger
# (above coordinate-rounding noise, far below real failures) are repaired by
# relaxing the cut offset out to the farthest known inner point.
CUT_GUARD_TRIGGER = 1e-4

# Dimension names of the projection-model variables. Postprocess cannot save
# dimensionless variables, so scalars get a trivial one-element dimension.
Z_DIM = "mga_z_axis"
_SCALAR_DIM = "mga_oracle_scalar_dim"


def _scalar_da(value):
    """A one-element DataArray on the trivial scalar dimension."""
    return xr.DataArray(np.array([value]), dims=_SCALAR_DIM,
                        coords={_SCALAR_DIM: [0]})


class MGA:
    """Near-optimal exploration on a solved ZEN-garden model.

    Holds the linopy model and the helpers shared by both modes. The split
    between `setup()` and the per-iteration methods exists because
    `add_constraints` is not idempotent: the near-optimality constraint is
    added exactly once, while the objective is swapped per iteration via
    `add_objective(..., overwrite=True)`. Oracle mode additionally adds an
    L-infinity projection model once (`setup_projection_model`).
    """

    # Dims aggregated away per axis kind, besides the member dim itself.
    _TECH_AGG = ["set_capacity_types", "set_location", "set_years"]
    _CARRIER_AGG = ["set_carriers", "set_nodes", "set_time_steps_operation"]

    def __init__(self, optimization_setup, epsilon, postprocess_ctx,
                 exclude_techs=None, include_techs=None,
                 include_carrier_imports=None, include_cost=False):
        """
        Args:
            optimization_setup: OptimizationSetup holding the solved baseline
                (model.objective.value is C*).
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Dict of Postprocess arguments for the iteration
                outputs (scenarios, subfolder, model_name, scenario_name,
                param_map).
            exclude_techs: Technologies dropped from the singleton tech axes.
                Mutually exclusive with include_techs.
            include_techs: Tech axes (names or {group: [members]} lumps).
            include_carrier_imports: Carrier-import axes, same format.
            include_cost: Add a normalised total-cost coordinate to the
                exploration space (oracle mode only).
        """
        if epsilon <= 0:
            raise ValueError(f"MGA epsilon must be positive, got {epsilon!r}")
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.epsilon = epsilon
        self.postprocess_ctx = postprocess_ctx
        self.include_cost = include_cost
        # Baseline objective C*, captured before MGA touches the model.
        self.c_star = self.model.objective.value

        self.cap_add = self.model.variables["capacity_addition"]
        # Selects the capacity type each axis aggregates per technology.
        self._capacity_mask = self._build_capacity_type_mask()

        # Normalisation state, set once by compute_fmax_normalization().
        self.u_star = None
        self._u_tilde = None
        self._offset = None
        # Known near-optimal points in explore coordinates; backs the
        # cut-validity guard in find_nearest_point (seeded with z*).
        self._inner_points = None

        all_techs = list(self.cap_add.coords["set_technologies"].values)
        if "flow_import" in self.model.variables:
            all_carriers = list(
                self.model.variables["flow_import"].coords["set_carriers"].values
            )
        else:
            all_carriers = []
        tech_groups, carrier_groups = build_axis_groups(
            include_techs, exclude_techs, include_carrier_imports,
            all_techs, all_carriers,
        )

        # The single source of truth for axis order everywhere downstream
        # (z, mu, name_list): tech axes first, then carrier axes.
        self.axes: list[Axis] = [
            Axis(name, TECH_CAPACITY, tuple(members),
                 self._selected_capacity_type(name, members))
            for name, members in tech_groups
        ] + [
            Axis(name, CARRIER_IMPORT, tuple(members), None)
            for name, members in carrier_groups
        ]
        self.z_names = [axis.name for axis in self.axes]
        self.n_z = len(self.axes)

        # Model handles needed by carrier-import axes.
        if any(axis.kind == CARRIER_IMPORT for axis in self.axes):
            self.flow_import = self.model.variables["flow_import"]
            self._ts_duration = (
                optimization_setup.parameters.time_steps_operation_duration
            )
        else:
            self.flow_import = None
            self._ts_duration = None

        # Baseline design vector z*, read now while the baseline solution is
        # still loaded (the fmax LPs overwrite it).
        self.z_star_raw = np.array(
            [self._axis_value(axis) for axis in self.axes], dtype=float
        )

        self._iter_count = 0

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        self.model.add_constraints(
            self._total_cost_expression() <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality",
        )
        logging.info(
            f"MGA: near-optimality constraint added, cost <= "
            f"{(1 + self.epsilon) * self.c_star} (C* = {self.c_star}, "
            f"epsilon = {self.epsilon})"
        )

    def _total_cost_expression(self):
        """The model's original total-cost objective as a linopy expression."""
        return self.optimization_setup.energy_system.rules.objective_total_cost(
            self.model
        )

    # ------------------------------------------------------------------
    # weights mode
    # ------------------------------------------------------------------

    def run_iteration(self, weights: dict, iter_id: int):
        """Solve one weights-mode iteration: min sum_i w_i * capacity_addition_i."""
        weight_array = self._build_weight_array(weights)
        self.model.add_objective(
            (weight_array * self.cap_add).sum(), sense="min", overwrite=True
        )
        logging.info(f"MGA iter {iter_id}: weights = {weights}")
        self._solve_and_postprocess(f"mga_iter_{iter_id}")

    def _build_weight_array(self, weights: dict) -> xr.DataArray:
        """Weights as a DataArray over the full set_technologies coordinate.

        Unlisted technologies get weight 0 (unknown names raise KeyError);
        (w * cap_add).sum() then aggregates the remaining dims by broadcasting.
        """
        tech_coord = self.cap_add.coords["set_technologies"]
        weight_array = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        for tech, weight in weights.items():
            weight_array.loc[tech] = float(weight)
        return weight_array

    # ------------------------------------------------------------------
    # oracle mode: axes on the model
    # ------------------------------------------------------------------

    @property
    def n_explore(self) -> int:
        """Dimension of the exploration space (n_z design axes + optional cost)."""
        return self.n_z + (1 if self.include_cost else 0)

    def _build_capacity_type_mask(self):
        """0/1 mask over (set_technologies, set_capacity_types): the capacity
        type each axis aggregates per technology.

        Storage technologies (more than one active capacity type) keep only
        their energy type; every other technology keeps its single power
        type. Active entries are detected from the live variable (labels !=
        -1), so no technology list is hardcoded; "power" is
        system.set_capacity_types[0] by ZEN-garden convention.
        """
        type_dim = "set_capacity_types"
        other_dims = [d for d in self.cap_add.dims
                      if d not in ("set_technologies", type_dim)]
        active = (self.cap_add.labels != -1).any(other_dims)
        power_type = str(self.optimization_setup.system.set_capacity_types[0])
        if power_type not in [str(c) for c in self.cap_add.coords[type_dim].values]:
            raise RuntimeError(
                f"MGA: power capacity type {power_type!r} not found in "
                f"capacity_addition."
            )
        is_storage = active.sum(type_dim) > 1
        is_power = active[type_dim] == power_type
        keep = active & ~(is_storage & is_power)
        storage_techs = [
            str(t) for t in active["set_technologies"].values
            if bool(is_storage.sel(set_technologies=t))
        ]
        logging.info(
            f"MGA capacity-type mask: storage tech(s) {storage_techs} use "
            f"energy capacity, all other techs their power capacity."
        )
        return keep.astype(float)

    def _selected_capacity_type(self, name: str, members: list) -> str:
        """The "+"-joined capacity type(s) the mask selects for one tech axis.

        Every member must map to the same selected type: lumping storage
        (energy, e.g. GWh) with non-storage (power, e.g. GW) members would
        mix incommensurable units.
        """
        selected = {
            member: tuple(
                str(c)
                for c in self._capacity_mask.coords["set_capacity_types"].values
                if float(self._capacity_mask.sel(set_technologies=member,
                                                 set_capacity_types=c)) > 0.5
            )
            for member in members
        }
        distinct = set(selected.values())
        if len(distinct) > 1:
            raise ValueError(
                f"MGA tech axis {name!r} mixes capacity types {selected}; "
                f"split storage and non-storage members into separate axes."
            )
        types = next(iter(distinct))
        if not types:
            raise RuntimeError(
                f"MGA tech axis {name!r}: no active capacity type for {members}."
            )
        return "+".join(types)

    def _axis_expression(self, axis: Axis, capacity, flow):
        """One axis value as an expression over `capacity`/`flow` data.

        `capacity` and `flow` are either the linopy variables (yielding the
        axis LinearExpression) or their `.solution` arrays (yielding the
        axis value): tech axes sum the capacity addition of the member
        technologies, restricted by the capacity-type mask; carrier axes sum
        the duration-weighted annual import of the member carriers,
        sum_{m,n,t} tau_t * flow[m, n, t].
        """
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            return (
                (self._capacity_mask * capacity)
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        return (
            (self._ts_duration * flow.sel(set_carriers=members))
            .sum(self._CARRIER_AGG)
        )

    def _axis_linexpr(self, axis: Axis):
        """Linopy expression of one axis (fmax objective, projection equality)."""
        return self._axis_expression(axis, self.cap_add, self.flow_import)

    def _axis_value(self, axis: Axis) -> float:
        """Value of one axis on the currently loaded solution."""
        flow = None if self.flow_import is None else self.flow_import.solution
        return float(self._axis_expression(axis, self.cap_add.solution, flow))

    def _to_explore_coords(self, z_design_raw, c_raw=None) -> np.ndarray:
        """Affine map from raw (design[, cost]) values to normalised ORACLE
        coordinates: z_i / U_i*, and (C - C*) / (epsilon * C*) for the cost
        axis."""
        assert self._u_tilde is not None, "compute_fmax_normalization() must run first"
        raw = np.asarray(z_design_raw, dtype=float)
        if self.include_cost:
            raw = np.append(raw, float(c_raw))
        return (raw - self._offset) / self._u_tilde

    def _extract_z(self) -> np.ndarray:
        """Exploration vector of the most recent solve, normalised, in
        canonical axis order (cost appended last when include_cost)."""
        z_design_raw = np.array(
            [self._axis_value(axis) for axis in self.axes], dtype=float
        )
        c_raw = None
        if self.include_cost:
            c_raw = float(self.model.variables["net_present_cost"].solution.sum())
        return self._to_explore_coords(z_design_raw, c_raw)

    @property
    def z_star_explore(self) -> np.ndarray:
        """Baseline point z* in normalised coordinates (design vector frozen
        in __init__; the baseline cost is exactly C*, so the cost coordinate
        is exactly 0)."""
        return self._to_explore_coords(self.z_star_raw, self.c_star)

    def polytope_metadata(self) -> dict:
        """Self-describing metadata for the saved polytope: per-axis kind,
        members, capacity type and physical unit, plus the normalisation
        convention. The augmented scale/offset are not stored -- they are an
        exact repackaging of (u_star, c_star, epsilon)."""
        units = self.optimization_setup.variables.units
        ureg = self.optimization_setup.energy_system.unit_handling.ureg
        axes_meta = [
            {
                "name": axis.name,
                "kind": axis.kind,
                "members": list(axis.members),
                "capacity_type": axis.capacity_type,
                "unit": axis_physical_unit(axis, units, ureg),
            }
            for axis in self.axes
        ]
        return {
            "axes": axes_meta,
            "cost_axis": "net_present_cost" if self.include_cost else None,
            "cost_unit": cost_physical_unit(units) if self.include_cost else None,
            "include_cost": bool(self.include_cost),
            "normalisation": (
                "design axis i: z_i / u_star[i]; "
                "cost axis (if present): (C - c_star) / (epsilon * c_star)"
            ),
        }

    # ------------------------------------------------------------------
    # oracle mode: normalisation, projection model, ORACLE callback
    # ------------------------------------------------------------------

    def compute_fmax_normalization(self) -> None:
        """Solve one LP per axis maximising the axis value over the
        near-optimal space.

        The maxima U_i* are both the normalisation denominators (z_i / U_i*)
        and the initial outer box. They are always solved fresh -- the LPs
        are cheap next to the full run, and caching would risk stale values.
        Each LP's full solution is saved via Postprocess as
        <model_name>_fmax_<axis>. Sets u_star and the augmented
        scale/offset; call after setup().
        """
        u_star = []
        for axis in self.axes:
            self.model.add_objective(
                self._axis_linexpr(axis), sense="max", overwrite=True
            )
            start = time.time()
            self.optimization_setup.solve()
            if not self.optimization_setup.optimality:
                raise RuntimeError(
                    f"MGA fmax LP for axis {axis.name!r} ended with "
                    f"{self.model.termination_condition!r} ('unbounded' means "
                    f"the axis has no finite near-optimal maximum)."
                )
            self._postprocess(f"fmax_{axis.name}")
            u_i = self._axis_value(axis)
            if not np.isfinite(u_i) or u_i <= 0:
                raise RuntimeError(
                    f"MGA fmax: U*[{axis.name}] = {u_i:.6g} cannot serve as a "
                    f"normalisation denominator; remove this axis from the config."
                )
            u_star.append(u_i)
            logging.info(
                f"MGA fmax: U*[{axis.name}] = {u_i:.6g} "
                f"(LP took {time.time() - start:.1f} s)"
            )
        self.u_star = np.array(u_star, dtype=float)

        # Augmented scale/offset: design axes (U_i*, 0); the cost axis, when
        # present, (epsilon * C*, C*).
        scale = list(self.u_star)
        offset = [0.0] * self.n_z
        if self.include_cost:
            scale.append(self.epsilon * self.c_star)
            offset.append(self.c_star)
        self._u_tilde = np.array(scale, dtype=float)
        self._offset = np.array(offset, dtype=float)

    def build_initial_outer_approximation(self) -> tuple[np.ndarray, np.ndarray]:
        """(A0, b0) of the initial outer polytope in normalised coordinates.

        Raw rows per design axis: -z_i <= 0 (axis values are sums of
        non-negative variables) and z_i <= U_i* (tight by construction);
        plus C* <= C <= (1 + epsilon) * C* when include_cost. Each raw row
        a^T z_raw <= b is then mapped to normalised coordinates via
        z_raw = offset + diag(u_tilde) z_norm, i.e. it becomes
        (a o u_tilde)^T z_norm <= b - a^T offset.
        """
        n_z = self.n_z
        A0 = np.vstack([-np.eye(n_z), np.eye(n_z)])
        b0 = np.concatenate([np.zeros(n_z), self.u_star])
        if self.include_cost:
            A0 = np.hstack([A0, np.zeros((A0.shape[0], 1))])
            cost_row = np.zeros((1, n_z + 1))
            cost_row[0, n_z] = 1.0
            A0 = np.vstack([A0, cost_row, -cost_row])
            b0 = np.concatenate(
                [b0, [(1.0 + self.epsilon) * self.c_star], [-self.c_star]]
            )

        b0 = b0 - A0 @ self._offset  # must precede the column scaling below
        A0 = A0 @ np.diag(self._u_tilde)

        # Sanity check: z* must satisfy the initial outer approximation.
        violation = A0 @ self.z_star_explore - b0
        if (violation > 1e-6 * (np.abs(b0) + 1.0)).any():
            raise RuntimeError(
                f"MGA: initial outer approximation excludes z* "
                f"(max violation {violation.max():.3g})."
            )
        logging.info(
            f"MGA outer approximation: {A0.shape[0]} rows, {A0.shape[1]} "
            f"normalised axes (include_cost = {self.include_cost})."
        )
        return A0, b0

    def _postprocess(self, label: str) -> None:
        """Write a Postprocess folder `<model_name>_<label>` for the
        currently loaded solution."""
        ctx = self.postprocess_ctx
        Postprocess(
            self.optimization_setup,
            scenarios=ctx["scenarios"],
            subfolder=ctx["subfolder"],
            model_name=f"{ctx['model_name']}_{label}",
            scenario_name=ctx["scenario_name"],
            param_map=ctx["param_map"],
        )

    def _solve_and_postprocess(self, label: str) -> None:
        """Solve the current model state and write its Postprocess folder."""
        self.optimization_setup.solve()
        if not self.optimization_setup.optimality:
            raise RuntimeError(
                f"MGA solve {label!r} failed: termination = "
                f"{self.model.termination_condition}"
            )
        self._postprocess(label)

    def setup_projection_model(self) -> None:
        """Add the L-infinity projection model to the linopy model; call once.

        Variables: delta (one entry per design axis on Z_DIM), a scalar t,
        and, with include_cost, a scalar delta_cost sharing t. Constraints:
        one projection equality per axis, axis_expr_i - delta_i == trial_i
        (RHS updated per iteration in find_nearest_point), and scaled
        t-bounds |delta_i| / U_i* <= t (plus |delta_cost| / (epsilon C*)
        <= t), so that min t is the normalised L-infinity distance to the
        trial point.
        """
        z_coord = xr.DataArray(
            np.array(self.z_names), dims=Z_DIM, coords={Z_DIM: self.z_names}
        )
        self.delta = self.model.add_variables(
            coords=[z_coord], name="mga_oracle_delta",
            lower=-np.inf, upper=np.inf,
        )
        self.t_var = self.model.add_variables(
            coords=[_scalar_da(0)], name="mga_oracle_t", lower=0.0,
        )

        for i, axis in enumerate(self.axes):
            self.model.add_constraints(
                self._axis_linexpr(axis) - self.delta.sel({Z_DIM: axis.name}) == 0.0,
                name=f"mga_oracle_proj_eq_axis{i}",
            )

        d_scale = xr.DataArray(
            1.0 / self.u_star, dims=Z_DIM, coords={Z_DIM: self.z_names}
        )
        self.model.add_constraints(
            d_scale * self.delta - self.t_var <= 0, name="mga_oracle_t_pos"
        )
        self.model.add_constraints(
            -(d_scale * self.delta) - self.t_var <= 0, name="mga_oracle_t_neg"
        )

        if self.include_cost:
            self.delta_cost = self.model.add_variables(
                coords=[_scalar_da(0)], name="mga_oracle_delta_cost",
                lower=-np.inf, upper=np.inf,
            )
            self.model.add_constraints(
                self._total_cost_expression() - self.delta_cost == _scalar_da(0.0),
                name="mga_oracle_proj_eq_cost",
            )
            c_scale = 1.0 / (self.epsilon * self.c_star)
            self.model.add_constraints(
                c_scale * self.delta_cost - self.t_var <= 0,
                name="mga_oracle_t_pos_cost",
            )
            self.model.add_constraints(
                -(c_scale * self.delta_cost) - self.t_var <= 0,
                name="mga_oracle_t_neg_cost",
            )

        # Seed the cut-validity guard with z*.
        self._inner_points = [np.asarray(self.z_star_explore, dtype=float)]
        logging.info(
            f"MGA oracle: projection model added (n_z = {self.n_z}, "
            f"include_cost = {self.include_cost})"
        )

    def find_nearest_point(self, trial_point: np.ndarray):
        """pyoNearOpt callback: project one trial point onto the near-optimal
        space and return it with its supporting cut.

        `trial_point` arrives in normalised coordinates (cost coordinate
        last when include_cost). Returns (z_feas, dist, mu_cut, b_cut, 0) in
        the same coordinates: dist = t* is the normalised L-infinity
        distance, mu_cut is the dual of the projection equalities rescaled
        into normalised coordinates (mu_raw o u_tilde; the affine cost
        offset cancels) and L2-normalised, and b_cut = mu_cut @ z_feas --
        possibly relaxed by the cut-validity guard (CUT_GUARD_TRIGGER).
        """
        assert trial_point.shape == (self.n_explore,), (
            f"trial point shape {trial_point.shape}, expected ({self.n_explore},)"
        )

        # Projection-equality RHS <- trial point in raw coordinates.
        trial_design_raw = trial_point[:self.n_z] * self.u_star
        for i in range(self.n_z):
            self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].rhs = float(
                trial_design_raw[i]
            )
        if self.include_cost:
            trial_c = float(trial_point[self.n_z])
            trial_c_raw = self.c_star + trial_c * self.epsilon * self.c_star
            self.model.constraints["mga_oracle_proj_eq_cost"].rhs = _scalar_da(
                trial_c_raw
            )

        # min t; .sum() collapses the trivial scalar dim for the objective.
        self.model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self._iter_count}"
        logging.info(
            f"MGA oracle: starting iteration {self._iter_count}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        try:
            self._solve_and_postprocess(label)
        except RuntimeError:
            # One numerically distressed projection must not kill a long run:
            # retry once (threaded barrier solves are not deterministic); on a
            # second failure return a zero-distance copy of the previous inner
            # point, which stops ORACLE gracefully via its identical-point
            # check, so artifacts and the final certificate still happen.
            logging.exception(
                f"MGA oracle iter {self._iter_count}: projection solve failed; "
                f"retrying once."
            )
            try:
                self._solve_and_postprocess(label)
            except RuntimeError:
                logging.error(
                    "MGA oracle: projection failed twice; stopping ORACLE "
                    "gracefully."
                )
                z_prev = np.asarray(self._inner_points[-1], dtype=float).copy()
                self._iter_count += 1
                return z_prev, 0.0, None, None, 0

        z_feas = self._extract_z()
        dist = float(self.t_var.solution.values[0])

        # Cut normal: duals of the projection equalities, rescaled into
        # normalised coordinates, then L2-normalised together with b_cut.
        mu_design_raw = np.array(
            [
                float(self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].dual.values)
                for i in range(self.n_z)
            ],
            dtype=float,
        )
        mu_cut = mu_design_raw * self.u_star
        if self.include_cost:
            mu_c_raw = float(
                self.model.constraints["mga_oracle_proj_eq_cost"].dual.values[0]
            )
            mu_cut = np.append(mu_cut, mu_c_raw * self.epsilon * self.c_star)
        scale = np.linalg.norm(mu_cut, ord=2)
        if scale > 1e-4:
            mu_cut = mu_cut / scale
        b_cut = float(mu_cut @ z_feas)

        # Cut-validity guard: a valid supporting hyperplane keeps every known
        # near-optimal point inside the outer approximation. If inexact duals
        # violate that, keep the direction but relax the offset out to the
        # farthest known inner point.
        if self._inner_points is not None:
            points = np.vstack(self._inner_points)
            worst = float((points @ mu_cut - b_cut).max())
            if worst > CUT_GUARD_TRIGGER:
                logging.warning(
                    f"MGA oracle iter {self._iter_count}: cut would remove "
                    f"known near-optimal point(s) by up to {worst:.4g}; "
                    f"relaxing b_cut {b_cut:.6g} -> {b_cut + worst:.6g}"
                )
                b_cut = float(b_cut + worst + 1e-9)
                if float(mu_cut @ trial_point) <= b_cut:
                    # The relaxed plane no longer separates the trial point;
                    # ORACLE may stall on a repeating trial point -- a loud
                    # stop, preferred over corrupting the outer approximation.
                    logging.error(
                        "MGA oracle: relaxed cut no longer excludes the trial "
                        "point (unreliable projection duals this iteration)."
                    )
            self._inner_points.append(np.asarray(z_feas, dtype=float))

        logging.info(
            f"MGA oracle iter {self._iter_count}: dist = {dist:.4g} "
            f"(normalised L-inf), |mu|_max = {np.max(np.abs(mu_cut)):.4g}, "
            f"b_cut = {b_cut:.4g}"
        )
        self._iter_count += 1
        return z_feas, dist, mu_cut, b_cut, 0


# ----------------------------------------------------------------------
# Event handler and mode dispatch
# ----------------------------------------------------------------------

@EventPublisher.register(Event.after_solve)
def run_mga(*, optimization_setup, scenarios, subfolder, model_name,
            scenario_name, param_map):
    """Run MGA after the baseline solve.

    Returns the oracle summary directory (oracle mode) or None.
    """
    mode = config["mode"]
    if mode not in ("weights", "oracle"):
        raise ValueError(f"Unknown MGA mode: {mode!r}. Expected 'weights' or 'oracle'.")
    if mode == "weights" and not config["iterations"]:
        logging.warning("MGA plugin: weights mode without iterations; skipping.")
        return None
    oracle_cfg = config["oracle"]

    logging.info(f"MGA plugin: mode = {mode!r}, epsilon = {config['epsilon']}")
    mga = MGA(
        optimization_setup,
        epsilon=config["epsilon"],
        postprocess_ctx={
            "scenarios": scenarios,
            "subfolder": subfolder,
            "model_name": model_name,
            "scenario_name": scenario_name,
            "param_map": param_map,
        },
        exclude_techs=config["exclude_techs"],
        include_techs=config["include_techs"],
        include_carrier_imports=config["include_carrier_imports"],
        include_cost=bool(oracle_cfg.get("include_cost", False)),
    )
    mga.setup()

    result = None
    if mode == "weights":
        for i, iteration in enumerate(config["iterations"]):
            mga.run_iteration(iteration["weights"], i)
    else:
        result = run_oracle_mode(mga, oracle_cfg)
    logging.info("MGA plugin: complete.")
    return result
