"""
MGA (Modeling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` event handler that, after the baseline solve, adds
a near-optimality cost constraint and re-solves under one of two modes:

    * "weights": user-provided list of per-iteration weight dicts; each
      iteration solves min sum_i w_i * cap_add_i.
    * "oracle": ORACLE algorithm (Turan, Moret, Bardow 2026). Iteratively
      refines inner/outer polytope approximations of Z_eps via L-infinity
      projections of trial points; the loop lives in pyoNearOpt.

For ORACLE, each exploration axis is either a tech-capacity axis (sum of
capacity_addition over member techs; singleton or lumped via `include_techs`)
or a carrier-import axis (duration-weighted annual flow_import over member
carriers, via `include_carrier_imports`). Coordinates are ALWAYS normalised:
design axis i is z_i / U_i* with U_i* from the per-axis fmax LPs (which also
provide the initial outer box, the unit cube), and the cost axis (when
`oracle.include_cost`) is (C - C*) / (eps * C*). The near-optimal cost budget
c^T x <= (1+eps)C* is enforced inside the model (added by setup()), not as a
polytope row. Every iteration is written to disk as a sibling sub-solution of
the baseline via Postprocess with a modified model_name.

Configuration (via the "plugins.mga" block in config.json):
    epsilon (float): near-optimality slack. Default 0.1.
    mode (str): "weights" | "oracle". Default "weights".
    exclude_techs (list[str]): techs dropped from the tech-capacity axes.
        oracle mode only. Mutually exclusive with include_techs.
    include_techs (list): positive list of tech-capacity axes. Entries are
        bare strings (singleton axis) or single-key dicts
        {group_name: [members]} (lumped axis), e.g.
        ["nuclear", {"ccs_lump": ["BF_BOF_CCS", "SMR_CCS"]}].
    include_carrier_imports (list): carrier-import axes, additive with the
        tech axes; same entry format, e.g. ["biomass"].
    iterations (list[dict]): weights mode only; one {"weights": {tech: w}}
        dict per iteration.
    oracle (dict): oracle mode only. max_iterations (int), tolerance (float,
        REQUIRED — no default), formulation (str, "kkt_milp" | "dual_bilinear",
        default "kkt_milp"): single-level reformulation of the step-2 max-min
        problem. "kkt_milp" is the published ORACLE MILP (complementarity via
        SOS1 or Big-M; its DUAL BOUND — the reported metric — stalls at the
        t_max cap once the inner point count grows, see bsc-thesis/code/diagnostics/ORACLE_metric_diagnosis_and_fix.md).
        "dual_bilinear" is the LP-duality reformulation (no binaries; solved
        globally by Gurobi spatial branching; NonConvex=2 is added to the
        solver options automatically) — same witness and metric semantics,
        but the bound actually closes. Md_override (float) and use_bigM
        (bool) apply to kkt_milp only; t_max_override (float|None) applies to
        both; include_cost (bool, default False); milp_options (dict of
        Gurobi options for ORACLE's internal step-2 solves — also used for
        the bilinear solve, which Gurobi treats as a MIP-like global solve,
        so MIPGap/MIPGapAbs/TimeLimit keep their meaning; defaults to
        DEFAULT_MILP_OPTIONS). final_certificate_time_limit (float seconds,
        default 0 = off): after a NON-converged loop, run one long step-2
        solve on the final geometry (capped by the loop's last reported
        metric) and store the improved certified metric in the npz — buys
        the proof once at the end instead of every iteration. Solver is
        hardcoded to Gurobi.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess

from .polytope_io import (
    CARRIER_IMPORT,
    TECH_CAPACITY,
    Polytope,
    save_polytope,
)


# Module-level config dict. The plugin loader merges user values from
# config.json's "plugins.mga" block via dict.update (SHALLOW). Therefore
# defaults for nested dicts like "oracle" must be applied at access time via
# .get(...), NOT relied upon to survive the loader merge.
config = {
    "epsilon": 0.1,
    "mode": "weights",
    "exclude_techs": [],
    "include_techs": [],
    "include_carrier_imports": [],
    "iterations": [],
    "oracle": {},
}

# Cut-validity guard trigger: a returned cut must keep every KNOWN
# near-optimal point (z* and all previous z_feas = pyoNearOpt's X) inside the
# outer approximation. Inexact projection duals (e.g. barrier without
# crossover, or a numerically distressed solve) can produce a plane that
# slices into R_eps: run-08's polytope row 391 cut off 14 inner points by up
# to 0.044. Violations above this trigger are repaired in find_nearest_point
# by relaxing b_cut out to the farthest known inner point. The trigger sits
# above coordinate-rounding noise (up to ~3e-4 from ORACLE's 1e-4 zeroing of
# z_feas after the cut is built) and far below real failures.
CUT_GUARD_TRIGGER = 1e-4

# Default Gurobi options for ORACLE's internal polytope MILPs; override via
# the oracle.milp_options config key (values must be numbers or strings).
DEFAULT_MILP_OPTIONS = {
    "OutputFlag": 1,
    "MIPGap": 0.05,
    "MIPGapAbs": 0.001,
    "TimeLimit": 1800,
    "Threads": 10,
}

# Coordinate dimension of the per-axis projection variables/constraints.
Z_DIM = "mga_z_axis"

# Substring needles for locating levels in the units-series MultiIndex (the
# level names vary slightly across ZEN-garden versions, e.g. "technology" vs
# "set_technologies"; see MGA._find_level).
_LEVEL_TECH = "technolog"
_LEVEL_CAPACITY_TYPE = "capacity_type"
_LEVEL_CARRIER = "carrier"


@dataclass(frozen=True)
class Axis:
    """One exploration axis of the MGA polytope.

    kind is TECH_CAPACITY (members are technologies; axis value = sum of
    capacity_addition over members, restricted to the selected capacity type)
    or CARRIER_IMPORT (members are carriers; axis value = duration-weighted
    annual flow_import over members). capacity_type is the "+"-joined selected
    capacity type(s) for tech axes and None for carrier axes.
    """

    name: str
    kind: str
    members: tuple[str, ...]
    capacity_type: str | None


def _parse_axis_entries(entries, valid_members, group_name_reserved,
                        seen_axis_names, label, member_noun,
                        member_noun_plural):
    """Parse one include_* config list into ordered (name, members) axis tuples.

    Shared by include_techs and include_carrier_imports. Each entry is either
    a bare string (singleton axis) or a single-key dict ``{name: [members]}``
    (lumped axis). Validates entry shape, axis-name uniqueness (against
    `seen_axis_names`, mutated in place), group-name collisions with the model
    names in `group_name_reserved` (dict entries only), and that each member
    appears at most once across the list. Unknown members (not in
    `valid_members`) are collected and raised as one KeyError at the end.
    Returns [(name, [members...]), ...] in user order.
    """
    parsed: list[tuple[str, list[str]]] = []
    unknown: list[str] = []
    seen_members: dict[str, str] = {}  # member -> axis that first claimed it
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            if not entry:
                raise ValueError(
                    f"MGA {label}[{idx}]: empty string is not a valid "
                    f"{member_noun} name."
                )
            name, members = entry, [entry]
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    f"MGA {label}[{idx}]: group dict must have exactly one "
                    f"key, got {len(entry)}: {sorted(entry.keys())}."
                )
            (name, members) = next(iter(entry.items()))
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"MGA {label}[{idx}]: group name must be a non-empty "
                    f"string, got {name!r}."
                )
            if name in group_name_reserved:
                raise ValueError(
                    f"MGA {label}[{idx}]: group name {name!r} collides with "
                    f"an existing technology or carrier name. Rename the "
                    f"group."
                )
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"MGA {label}[{idx}] ({name!r}): group members must be a "
                    f"non-empty list of {member_noun} names, got {members!r}."
                )
            if not all(isinstance(m, str) and m for m in members):
                raise ValueError(
                    f"MGA {label}[{idx}] ({name!r}): every member must be a "
                    f"non-empty string, got {members!r}."
                )
        else:
            raise ValueError(
                f"MGA {label}[{idx}]: entry must be a string or a single-key "
                f"dict, got {type(entry).__name__}: {entry!r}."
            )
        if name in seen_axis_names:
            raise ValueError(
                f"MGA {label}[{idx}]: duplicate axis name {name!r}."
            )
        seen_axis_names.add(name)
        for m in members:
            if m not in valid_members:
                unknown.append(m)
            elif m in seen_members:
                raise ValueError(
                    f"MGA {label}[{idx}] ({name!r}): {member_noun} {m!r} also "
                    f"appears in axis {seen_members[m]!r}; each {member_noun} "
                    f"may appear at most once."
                )
            else:
                seen_members[m] = name
        parsed.append((name, list(members)))
    if unknown:
        raise KeyError(
            f"MGA {label} contains unknown {member_noun_plural}: "
            f"{sorted(set(unknown))}"
        )
    return parsed


class MGA:
    """MGA core logic.

    Holds the linopy model and helper methods shared by the two exploration
    modes. The split between `setup()` and the per-iteration methods exists
    because `linopy.Model.add_constraints` is not idempotent: the
    near-optimality constraint is added exactly once in `setup()`, while the
    objective is swapped per iteration via `add_objective(..., overwrite=True)`.

    For `oracle` mode, an additional projection-model variable (delta), a
    scalar t, and three constraints are added by `setup_projection_model()`.
    """

    def __init__(self, optimization_setup, epsilon, postprocess_ctx, exclude_techs=None,
                 include_techs=None, include_carrier_imports=None, include_cost=False):
        """
        Args:
            optimization_setup: OptimizationSetup holding the solved baseline
                (so model.objective.value equals C*).
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Dict forwarded to Postprocess for each iteration's
                output (scenarios, subfolder, model_name, scenario_name,
                param_map).
            exclude_techs: Tech names dropped from the tech-capacity axes.
                Mutually exclusive with include_techs.
            include_techs: Positive list of tech-capacity axes (bare string or
                ``{name: [members]}`` lump; see module docstring).
            include_carrier_imports: Carrier-import axes, additive with the
                tech axes; same entry format.
            include_cost: If True, augment z with a total-cost coordinate
                (oracle mode only).
        """
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.cap_add = self.model.variables["capacity_addition"]
        # Capacity-type mask (storage fix): keeps, per tech, the energy type
        # for storage techs and the single power type otherwise. Applied
        # wherever an axis aggregates cap_add over set_capacity_types, so the
        # fmax LP, baseline z*, projection equality, and z-extraction all agree.
        self._cap_keep = self._build_capacity_keep_mask()
        if epsilon <= 0:
            raise ValueError(f"MGA epsilon must be positive, got {epsilon!r}")
        self.epsilon = epsilon
        # C* captured here, before any MGA modifications to model.objective.
        self.c_star = self.model.objective.value
        self.postprocess_ctx = postprocess_ctx

        self.include_cost = include_cost
        # fmax state, populated by compute_fmax_normalization():
        #   u_star   - per-axis U_g* on z_names order
        #   _u_tilde - augmented scale (u_star, [eps*C*]) on explore order
        #   _offset  - augmented offset (0..., [C*]) on explore order
        # ORACLE always normalises; these must be set before any coordinate
        # transform (see _to_explore_coords, which raises if they are None).
        self.u_star = None
        self._u_tilde = None
        self._offset = None

        # Cut-validity guard state (oracle mode): every point handed to the
        # inner approximation, in explore coordinates. Seeded with z* by
        # setup_projection_model, extended and checked against each new cut
        # in find_nearest_point.
        self._inner_points = None

        # --- Build the canonical list of exploration axes (see class Axis) ---
        # Tech axes come first (in the order implied by include_techs /
        # exclude_techs), then carrier axes (user order).
        all_techs = list(self.cap_add.coords["set_technologies"].values)
        all_techs_set = set(all_techs)
        # Carriers (for carrier-import axes + the group-name reserved set).
        if "flow_import" in self.model.variables:
            all_carriers = list(
                self.model.variables["flow_import"].coords["set_carriers"].values
            )
        else:
            all_carriers = []
        all_carriers_set = set(all_carriers)

        include_techs_list = list(include_techs or [])
        exclude_techs_list = list(exclude_techs or [])
        include_carrier_list = list(include_carrier_imports or [])

        if include_techs_list and exclude_techs_list:
            raise ValueError(
                "MGA: include_techs and exclude_techs are mutually exclusive "
                "(both non-empty)."
            )

        if include_techs_list:
            # include_techs path: axis order follows USER order (not model
            # order) — intentional; no downstream consumer keys off model order.
            z_groups = _parse_axis_entries(
                include_techs_list,
                valid_members=all_techs_set,
                group_name_reserved=all_techs_set,
                seen_axis_names=set(),
                label="include_techs",
                member_noun="tech",
                member_noun_plural="technologies",
            )
        else:
            # Legacy exclude_techs path: singleton axes in MODEL order.
            excluded_set = set(exclude_techs_list)
            unknown = excluded_set - all_techs_set
            if unknown:
                raise KeyError(
                    f"exclude_techs contains unknown technologies: "
                    f"{sorted(unknown)}"
                )
            z_groups = [(t, [t]) for t in all_techs if t not in excluded_set]

        # --- Carrier-import axes (additive; orthogonal to tech axes) ---
        # Group/lump names must be a fresh label, not an existing tech OR
        # carrier name (the polytope name_list shares one namespace).
        reserved_names = all_techs_set | all_carriers_set
        carrier_groups = _parse_axis_entries(
            include_carrier_list,
            valid_members=all_carriers_set,
            group_name_reserved=reserved_names,
            seen_axis_names={name for name, _ in z_groups},
            label="include_carrier_imports",
            member_noun="carrier",
            member_noun_plural="carriers",
        )

        # self.axes is the SINGLE source of truth for axis ordering everywhere
        # downstream (z, mu, name_list): tech axes first, then carrier axes.
        # The capacity type per tech axis is computed here (storage fix: every
        # member must share the same selected type, see
        # _compute_axis_capacity_type).
        self.axes: list[Axis] = [
            Axis(name, TECH_CAPACITY, tuple(members),
                 self._compute_axis_capacity_type(name, members))
            for name, members in z_groups
        ] + [
            Axis(name, CARRIER_IMPORT, tuple(members), None)
            for name, members in carrier_groups
        ]
        self.z_names: list[str] = [a.name for a in self.axes]
        self.n_z: int = len(self.axes)
        self.has_carrier_axis: bool = any(
            a.kind == CARRIER_IMPORT for a in self.axes
        )

        # Model handles for carrier-import axes (looked up once).
        if self.has_carrier_axis:
            self.flow_import = self.model.variables["flow_import"]
            # Operational time-step duration tau_t (set_time_steps_operation).
            self._ts_duration = (
                self.optimization_setup.parameters.time_steps_operation_duration
            )
        else:
            self.flow_import = None
            self._ts_duration = None

        # Baseline design vector z*, captured NOW while the baseline solution is
        # still loaded (the fmax LPs overwrite it). Per-axis, kind-aware.
        self._z_star_design_raw = np.array(
            [self._axis_value(a) for a in self.axes], dtype=float
        )

        # Iteration counter for labelling Postprocess folders.
        self._iter_count = 0

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        orig_cost_expr = self.optimization_setup.energy_system.rules.objective_total_cost(self.model)
        self.model.add_constraints(
            orig_cost_expr <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality",
        )
        logging.info(
            f"MGA: added near-optimality constraint cost <= "
            f"{(1 + self.epsilon) * self.c_star} "
            f"(C* = {self.c_star}, epsilon = {self.epsilon})"
        )

    # ------------------------------------------------------------------
    # weights-mode methods
    # ------------------------------------------------------------------

    def run_iteration(self, weights: dict, iter_id: int):
        """Run one weights-mode MGA iteration."""
        w = self._build_w(weights)
        mga_obj = (w * self.cap_add).sum()
        self.model.add_objective(mga_obj, sense="min", overwrite=True)
        logging.info(f"MGA iter {iter_id}: objective replaced with weights = {weights}")

        self._solve_and_postprocess(f"mga_iter_{iter_id}")

    def _build_w(self, weights: dict) -> xr.DataArray:
        """Build a 1D DataArray over the FULL set_technologies coord from a dict.

        Used by weights mode. Unknown technology names raise KeyError.
        Missing technologies default to weight 0; (w * cap_add).sum()
        implicitly aggregates over (cap_type, location, year) via xarray
        broadcasting.
        """
        tech_coord = self.cap_add.coords["set_technologies"]
        w = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        known = set(tech_coord.values)
        for tech, val in weights.items():
            if tech not in known:
                raise KeyError(f"Unknown technology in MGA weights: {tech!r}")
            w.loc[tech] = float(val)
        return w

    # ------------------------------------------------------------------
    # Common helpers used by oracle mode
    # ------------------------------------------------------------------

    @property
    def n_explore(self) -> int:
        """Dimension of the exploration vector handed to ORACLE (n_z, or n_z+1
        when include_cost augments z with a cost coordinate)."""
        return self.n_z + (1 if self.include_cost else 0)

    # Aggregation dims per axis kind (module-private constants inlined here).
    _TECH_AGG = ["set_capacity_types", "set_location", "set_years"]
    _CARRIER_AGG = ["set_carriers", "set_nodes", "set_time_steps_operation"]

    def _build_capacity_keep_mask(self):
        """0/1 coefficient mask over (set_technologies, set_capacity_types).

        Storage technologies (more than one active capacity type) keep ONLY
        their energy capacity; every other technology keeps its single (power)
        type. Detected from the live capacity_addition variable (``labels !=
        -1`` marks an active entry), so no tech list is hardcoded. ``power``
        is ``system.set_capacity_types[0]`` — ZEN-garden's convention for the
        single type of non-storage techs; the energy type(s) are the rest.
        Returns the 0/1 mask as a float DataArray.
        """
        ct_dim = "set_capacity_types"
        extra = [d for d in self.cap_add.dims if d not in ("set_technologies", ct_dim)]
        # active[tech, captype]: an actual variable exists for some (loc, year)
        active = (self.cap_add.labels != -1).any(extra)
        cap_types = [str(c) for c in self.cap_add.coords[ct_dim].values]
        power_ct = str(self.optimization_setup.system.set_capacity_types[0])
        if power_ct not in cap_types:
            raise RuntimeError(
                f"MGA storage fix: power capacity type {power_ct!r} "
                f"(system.set_capacity_types[0]) is not among capacity_addition's "
                f"capacity types {cap_types}."
            )
        energy_types = [c for c in cap_types if c != power_ct]
        n_active = active.sum(ct_dim)            # per tech: 1 (power only) or >1 (storage)
        is_power = active[ct_dim] == power_ct    # along set_capacity_types
        multi = n_active > 1                     # storage techs
        # Keep every active entry EXCEPT the power entry of multi-type techs.
        keep = active & ~(multi & is_power)
        multi_techs = [
            str(t) for t in active["set_technologies"].values
            if bool(multi.sel(set_technologies=t))
        ]
        logging.info(
            f"MGA storage fix: power capacity type = {power_ct!r}, energy "
            f"capacity type(s) = {energy_types}; {len(multi_techs)} multi-"
            f"capacity-type (storage) tech(s) will use ENERGY only: {multi_techs}."
        )
        return keep.astype(float)

    def _compute_axis_capacity_type(self, name: str, members: list[str]) -> str:
        """Selected capacity type of one tech axis (storage fix).

        After the capacity-type mask, each tech axis aggregates ONE capacity
        quantity per member: energy for storage techs, power otherwise. A
        lumped tech axis mixing storage (energy/GWh) with non-storage
        (power/GW) members would re-introduce unit mixing, so every member
        must share the same selected capacity type(s). Returns the "+"-joined
        selected type (for the polytope metadata).
        """
        sigs = {
            m: tuple(
                str(c)
                for c in self._cap_keep.coords["set_capacity_types"].values
                if float(self._cap_keep.sel(set_technologies=m,
                                            set_capacity_types=c)) > 0.5
            )
            for m in members
        }
        distinct = set(sigs.values())
        if len(distinct) > 1:
            raise ValueError(
                f"MGA tech axis {name!r}: members map to different capacity "
                f"types {sigs}. Lumping storage (energy) with non-storage "
                f"(power) techs mixes incommensurable units (GWh vs GW); "
                f"split them into separate axes."
            )
        sig = next(iter(distinct))
        if not sig:
            raise RuntimeError(
                f"MGA tech axis {name!r}: no active capacity type for "
                f"members {members}; cannot form an axis."
            )
        return "+".join(sig)

    def _axis_linexpr(self, axis: Axis):
        """linopy LinearExpression for one axis (fmax LP objective / projection).

        Branches on axis kind (the ONLY place, besides _axis_value, that does):
          TECH_CAPACITY  -> sum of capacity_addition over the axis's member
                            techs and over (cap_type, loc, year), restricted by
                            _cap_keep to the selected capacity type per tech
                            (energy for storage techs, power otherwise).
          CARRIER_IMPORT -> duration-weighted sum of flow_import over the axis's
                            member carriers, nodes, and operational time steps:
                            sum_{m,n,t} tau_t * flow_import[m, n, t].
        """
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            return (
                (self._cap_keep * self.cap_add)
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        # CARRIER_IMPORT
        return (
            (self._ts_duration * self.flow_import.sel(set_carriers=members))
            .sum(self._CARRIER_AGG)
        )

    def _axis_value(self, axis: Axis) -> float:
        """Scalar value of one axis on the currently loaded solution.

        Same expression as _axis_linexpr but evaluated on `.solution`; used for
        the frozen baseline z* and for _extract_z after each projection solve.
        """
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            return float(
                (self._cap_keep * self.cap_add.solution)
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        # CARRIER_IMPORT
        return float(
            (self._ts_duration * self.flow_import.solution.sel(set_carriers=members))
            .sum(self._CARRIER_AGG)
        )

    def _to_explore_coords(self, z_design_raw: np.ndarray, c_raw=None) -> np.ndarray:
        """Map a raw (design, cost) solution to ORACLE polytope coordinates.

        ORACLE always normalises: this applies the affine map
        (z - offset) / U_tilde, so design axes become z_i/U_i* and (when
        include_cost) the cost axis becomes (C - C*)/(eps*C*). The fmax LPs
        must have run first (u_star/_u_tilde/_offset populated).
        """
        if self._u_tilde is None:
            raise RuntimeError(
                "MGA: coordinate normalisation requested before the fmax LPs "
                "ran; compute_fmax_normalization() must be called first."
            )
        raw = np.asarray(z_design_raw, dtype=float)
        if self.include_cost:
            raw = np.append(raw, float(c_raw))
        return (raw - self._offset) / self._u_tilde

    def _extract_z(self) -> np.ndarray:
        """Read the exploration vector z from the most recent solve.

        Returns a 1D ndarray of length n_explore in the canonical axes order
        (cost appended last when include_cost), in normalised coordinates.
        Each design entry is its axis value (kind-aware, see _axis_value).
        """
        z_design_raw = np.array(
            [self._axis_value(a) for a in self.axes], dtype=float
        )
        c_raw = None
        if self.include_cost:
            c_raw = float(self.model.variables["net_present_cost"].solution.sum())
        return self._to_explore_coords(z_design_raw, c_raw)

    @property
    def z_star_explore(self) -> np.ndarray:
        """Baseline point z* in ORACLE polytope coordinates.

        Uses the baseline design vector frozen in __init__ (the fmax LPs
        overwrite the loaded solution). The baseline cost is exactly C*, so the
        cost coordinate (when include_cost) is exactly 0.
        """
        return self._to_explore_coords(self._z_star_design_raw, self.c_star)

    @staticmethod
    def _find_level(index, needle: str):
        """Name of the first index level containing `needle` (e.g. 'technolog',
        'capacity_type', 'carrier'), or None."""
        return next((lvl for lvl in index.names if needle in lvl), None)

    def _axis_physical_unit(self, axis: Axis):
        """Original physical unit string of the axis VALUE, read from the
        model's unit handling (same source as var_dict.h5's *_units). None
        when unit tracking is off (solver.check_unit_consistency=False) or
        unavailable.

          tech axis    -> capacity_addition unit at the selected capacity type
                          (energy types are already annualised by ZEN-garden).
          carrier axis -> flow_import is an instantaneous (power) unit; the
                          axis is the duration-weighted ANNUAL import, so the
                          unit is "<flow unit> * hour" (energy).

        Heterogeneous lumps yield a ' + '-joined string.
        """
        units = self.optimization_setup.variables.units
        members = list(axis.members)
        if axis.kind == TECH_CAPACITY:
            ser = units.get("capacity_addition")
            if ser is None:
                return None
            selected = axis.capacity_type.split("+")
            tech_lvl = self._find_level(ser.index, _LEVEL_TECH)
            ct_lvl = self._find_level(ser.index, _LEVEL_CAPACITY_TYPE)
            if tech_lvl is None or ct_lvl is None:
                return None
            mask = (
                ser.index.get_level_values(tech_lvl).isin(members)
                & ser.index.get_level_values(ct_lvl).isin(selected)
            )
            vals = sorted({str(u) for u in ser[mask].to_numpy()})
            return " + ".join(vals) if vals else None
        # CARRIER_IMPORT: annualise the instantaneous flow unit (× hour).
        ser = units.get("flow_import")
        if ser is None:
            return None
        carrier_lvl = self._find_level(ser.index, _LEVEL_CARRIER)
        if carrier_lvl is None:
            return None
        mask = ser.index.get_level_values(carrier_lvl).isin(members)
        ureg = self.optimization_setup.energy_system.unit_handling.ureg
        annual = set()
        for u in {str(x) for x in ser[mask].to_numpy()}:
            try:
                annual.add(str(ureg(f"({u}) * hour").units))
            except Exception:
                annual.add(f"({u}) * hour")
        return " + ".join(sorted(annual)) if annual else None

    def _cost_physical_unit(self):
        """Physical unit of net_present_cost (e.g. 'megaEuro'), or None."""
        ser = self.optimization_setup.variables.units.get("net_present_cost")
        if ser is None:
            return None
        vals = sorted({str(u) for u in np.atleast_1d(np.asarray(ser))})
        return " + ".join(vals) if vals else None

    def polytope_metadata(self) -> dict:
        """Self-describing metadata for the saved polytope.

        Everything a consumer needs to de-normalise and interpret polytope.npz
        without the config: per design axis (z_names order) its kind, members,
        selected capacity_type (tech axes only) and physical `unit` string
        (null if unit tracking is off); the cost axis unit in `cost_unit`.
        u_tilde/offset are NOT stored — with the fixed normalisation
        convention (z_i / U_i*; cost (C - C*) / (eps * C*)) they are an exact
        repackaging of (u_star, c_star, epsilon):
            u_tilde = [u_star..., eps*C*],  offset = [0..., C*]
        """
        axes = [
            {
                "name": a.name,
                "kind": a.kind,
                "members": list(a.members),
                "capacity_type": a.capacity_type,
                "unit": self._axis_physical_unit(a),
            }
            for a in self.axes
        ]
        return {
            "axes": axes,
            "cost_axis": "net_present_cost" if self.include_cost else None,
            "cost_unit": self._cost_physical_unit() if self.include_cost else None,
            "include_cost": bool(self.include_cost),
            "normalisation": (
                "design axis i: z_i / u_star[i]; "
                "cost axis (if present): (C - c_star) / (epsilon * c_star)"
            ),
        }

    def compute_fmax_normalization(self) -> None:
        """Compute U_g* = max z_g over the near-optimal polytope, per z-axis.

        One LP per axis: baseline model state plus the near-optimality cost
        cap (added by setup()), objective replaced by max(axis value). U_g*
        serves dually as the per-axis upper bound of the initial outer
        approximation and as the normalisation denominator z_g / U_g*.

        Every LP is solved on every run — NO caching (the LPs are cheap
        relative to total runtime, and always solving removes a class of
        silent cache-staleness errors). Each LP's full solution is persisted
        via Postprocess as ``<model_name>_fmax_<axis>``.

        Sets self.u_star, self._u_tilde, self._offset. Call once, after
        setup() and before setup_projection_model().
        """
        # --- LP loop: maximise every axis over the near-optimal space ---
        u_values: dict[str, float] = {}
        for axis in self.axes:
            members = list(axis.members)
            obj = self._axis_linexpr(axis)
            self.model.add_objective(obj, sense="max", overwrite=True)
            t0 = time.time()
            self.optimization_setup.solve()
            elapsed = time.time() - t0
            if not self.optimization_setup.optimality:
                raise RuntimeError(
                    f"MGA oracle fmax: LP for axis {axis.name!r} ({axis.kind}, "
                    f"members {members}) did not solve to optimality "
                    f"(termination = {self.model.termination_condition!r}). An "
                    f"'unbounded' status means this axis has no finite "
                    f"near-optimal maximum — exclude the offending member."
                )
            # Persist the full extreme near-optimal design for this axis using
            # the same Postprocess machinery as the per-iteration solves.
            self._postprocess(f"fmax_{axis.name}")
            u_i = self._axis_value(axis)
            # Fail fast: a zero/negative U_g* cannot serve as a normalisation
            # denominator, and waiting for the remaining LPs wastes a run.
            if not np.isfinite(u_i) or u_i <= 0:
                key = ("include_carrier_imports" if axis.kind == CARRIER_IMPORT
                       else "include_techs / exclude_techs")
                raise RuntimeError(
                    f"MGA oracle fmax: U*[{axis.name}] = {u_i:.6g} "
                    f"({axis.kind}, members {members}) — cannot normalise. "
                    f"Remove this axis (or the offending member) via "
                    f"plugins.mga.{key}."
                )
            u_values[axis.name] = u_i
            logging.info(
                f"MGA oracle fmax: U*[{axis.name}] = {u_i:.6g} ({axis.kind}, "
                f"LP took {elapsed:.1f} s, {len(members)} "
                f"member{'s' if len(members) != 1 else ''}; full design saved)."
            )

        self.u_star = np.array(
            [u_values[n] for n in self.z_names], dtype=float
        )

        # Augmented scale/offset: design axes use U_i* with offset 0; the cost
        # axis (when include_cost) uses the slack range eps*C* with offset C*.
        u = list(self.u_star)
        off = [0.0] * self.n_z
        if self.include_cost:
            u.append(self.epsilon * self.c_star)
            off.append(self.c_star)
        self._u_tilde = np.array(u, dtype=float)
        self._offset = np.array(off, dtype=float)
        logging.info(
            f"MGA oracle fmax: ready. U* range [{self.u_star.min():.4g}, "
            f"{self.u_star.max():.4g}] (raw z units); "
            f"cost axis scale eps*C* = {self.epsilon * self.c_star:.6g}"
            + (" (active)" if self.include_cost else " (unused, include_cost=False)")
            + "."
        )

    def build_initial_outer_approximation(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct (A0, b0) for the initial outer polytope, in axes order.

        Row blocks (n_z axes; bounds come solely from the fmax maxima U_i*):
          1. n_z non-negativity rows  -z_i <= 0  (every axis value is a sum of
             non-negative variables).
          2. n_z fmax upper bounds     z_i <= U_i*  (valid and tight: U_i* is
             by construction the max of z_i over the near-optimal space).
          3. include_cost only: C <= (1+eps)*C* and -C <= -C*.
        The near-optimal cost budget itself lives in the energy-system model
        (added once by setup()), so it binds every projection solve regardless
        of these rows.

        The raw polytope is then mapped to normalised coordinates: with
        z_raw = offset + diag(U_tilde) z_norm, each raw row a^T z_raw <= b
        becomes (a o U_tilde)^T z_norm <= b - a^T offset; the fmax rows become
        z_norm,i <= 1. Containment of z* is asserted before returning.

        Returns:
            (A0, b0): np.ndarray of shape (n_rows, n_explore) and (n_rows,),
            in ORACLE (normalised) polytope coordinates.
        """
        n_z = self.n_z

        # --- Raw design rows: non-negativity + fmax upper bounds ---
        #   -z_i <= 0           (non-negativity)
        #    z_i <= U_i*        (fmax box)
        A0 = np.vstack([-np.eye(n_z), np.eye(n_z)])
        b0 = np.concatenate([np.zeros(n_z), self.u_star])

        # --- Cost coordinate: 2 rows  C <= (1+eps)C*  and  -C <= -C*  ---
        if self.include_cost:
            # widen every existing row with a zero cost column
            A0 = np.hstack([A0, np.zeros((A0.shape[0], 1))])
            cost_up = np.zeros((1, n_z + 1)); cost_up[0, n_z] = 1.0
            cost_lo = np.zeros((1, n_z + 1)); cost_lo[0, n_z] = -1.0
            A0 = np.vstack([A0, cost_up, cost_lo])
            b0 = np.concatenate(
                [b0, [(1.0 + self.epsilon) * self.c_star], [-self.c_star]]
            )

        # --- Normalisation transform (always on for ORACLE) ---
        # Each raw row  a^T z_raw <= b  with z_raw = offset + diag(U_tilde) z_norm
        # becomes  (a o U_tilde)^T z_norm <= b - a^T offset.
        b0 = b0 - A0 @ self._offset       # uses raw A0 — must precede scaling
        A0 = A0 @ np.diag(self._u_tilde)

        # --- Sanity assert: A0 @ z*_explore <= b0  (containment) ---
        z_star = self.z_star_explore
        lhs = A0 @ z_star
        abs_tol = 1e-6 * (np.abs(b0) + 1.0)
        violations = lhs > b0 + abs_tol
        if violations.any():
            i = int(np.argmax(lhs - b0))
            raise RuntimeError(
                f"MGA initial outer approximation violates containment of z*: "
                f"row {i}: lhs={lhs[i]:.6g} > rhs={b0[i]:.6g} "
                f"(slack {lhs[i] - b0[i]:.3g}). Row blocks: [0,{n_z}) "
                f"non-negativity, [{n_z},{2 * n_z}) fmax upper bounds, "
                f"remaining rows cost upper/lower."
            )

        # --- Diagnostic logging ---
        n_cost_rows = 2 if self.include_cost else 0
        n_tech = sum(1 for a in self.axes if a.kind == TECH_CAPACITY)
        n_carrier = self.n_z - n_tech
        logging.info(
            f"MGA outer approximation [include_cost={self.include_cost}]: "
            f"n_explore={A0.shape[1]} (n_z={n_z} axes = {n_tech} tech + "
            f"{n_carrier} carrier{', + 1 cost' if self.include_cost else ''}), "
            f"rows={A0.shape[0]} = {n_z} non-neg + {n_z} fmax upper "
            f"+ {n_cost_rows} cost. Coordinates NORMALISED (z/U*): design axes "
            f"dimensionless with near-optimal range [0,1]"
            + ("; cost axis = (C-C*)/(eps*C*), range [0,1]." if self.include_cost
               else ".")
        )
        return A0, b0

    def _postprocess(self, label: str) -> None:
        """Write a Postprocess folder labelled `<base>_<label>` for the currently
        loaded solution. Shared by the per-iteration solves and the fmax LPs."""
        base_name = self.postprocess_ctx["model_name"]
        Postprocess(
            self.optimization_setup,
            scenarios=self.postprocess_ctx["scenarios"],
            subfolder=self.postprocess_ctx["subfolder"],
            model_name=f"{base_name}_{label}",
            scenario_name=self.postprocess_ctx["scenario_name"],
            param_map=self.postprocess_ctx["param_map"],
        )

    def _solve_and_postprocess(self, label: str) -> None:
        """Solve current model state and write a Postprocess folder labelled `<base>_<label>`."""
        self.optimization_setup.solve()
        if not self.optimization_setup.optimality:
            raise RuntimeError(
                f"MGA solve failed (label={label}): "
                f"termination = {self.model.termination_condition}"
            )
        self._postprocess(label)

    # ------------------------------------------------------------------
    # oracle-mode setup and callback
    # ------------------------------------------------------------------

    def setup_projection_model(self) -> None:
        """Add the L-infinity projection model to the linopy model.

        One scalar projection equality per axis, mga_oracle_proj_eq_axis{i}:
        axis_expr_i - delta[i] == trial_i, with the kind-aware _axis_linexpr.
        The RHS is updated per iteration in find_nearest_point. delta is a
        vector on the Z_DIM dim so the t-constraints stay vectorised.

        Coordinates are ALWAYS normalised: d_scale_g = 1/U_g* in the t-bounds,
        so min t = max_g |delta_g|/U_g* = the normalised L-inf distance.
        Variables: mga_oracle_delta (vector on Z_DIM), mga_oracle_t (scalar),
        and mga_oracle_delta_cost (scalar, include_cost only; shares t_var,
        c_scale = 1/(eps*C*)).

        Call exactly once before the ORACLE loop.
        """
        z_coord = xr.DataArray(
            np.array(self.z_names),
            dims=Z_DIM,
            coords={Z_DIM: self.z_names},
        )

        self.delta = self.model.add_variables(
            coords=[z_coord], name="mga_oracle_delta",
            lower=-np.inf, upper=np.inf,
        )

        # t_var needs a (trivial) dimension so that ZEN-garden's Postprocess
        # can convert it to a DataFrame. A truly scalar variable would crash
        # postprocess.save_var with "cannot convert a scalar to a DataFrame".
        t_coord = xr.DataArray(
            np.array([0]),
            dims="mga_oracle_scalar_dim",
            coords={"mga_oracle_scalar_dim": [0]},
        )
        self.t_var = self.model.add_variables(
            coords=[t_coord], name="mga_oracle_t",
            lower=0.0,
        )

        # One scalar projection equality per axis; the RHS (trial point) is
        # updated each iteration in find_nearest_point.
        for i, axis in enumerate(self.axes):
            expr = self._axis_linexpr(axis)
            self.model.add_constraints(
                expr - self.delta.sel({Z_DIM: axis.name}) == 0.0,
                name=f"mga_oracle_proj_eq_axis{i}",
            )

        # Per-axis L-inf scaling: d_scale_g = 1/U_g* (normalisation always on),
        # so min t yields t* = max_g |delta_g|/U_g* = the normalised L-inf
        # distance. Vectorised over the whole delta vector.
        d_scale = xr.DataArray(
            1.0 / self.u_star,
            dims=Z_DIM,
            coords={Z_DIM: self.z_names},
        )
        self.model.add_constraints(
            d_scale * self.delta - self.t_var <= 0, name="mga_oracle_t_pos"
        )
        self.model.add_constraints(
            -(d_scale * self.delta) - self.t_var <= 0, name="mga_oracle_t_neg"
        )

        # --- Cost-coordinate projection track (include_cost only) ---
        # delta_cost absorbs C(x) - trial_C; the SHARED t_var makes t* the
        # L-inf distance over all n_z+1 normalised axes at once.
        if self.include_cost:
            self.delta_cost = self.model.add_variables(
                coords=[t_coord], name="mga_oracle_delta_cost",
                lower=-np.inf, upper=np.inf,
            )
            cost_expr = self.optimization_setup.energy_system.rules.objective_total_cost(
                self.model
            )
            zero_cost = xr.DataArray(
                np.zeros(1),
                dims="mga_oracle_scalar_dim",
                coords={"mga_oracle_scalar_dim": [0]},
            )
            self.model.add_constraints(
                cost_expr - self.delta_cost == zero_cost,
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

        # Seed the cut-validity guard with z* (row 0 of pyoNearOpt's X).
        self._inner_points = [np.asarray(self.z_star_explore, dtype=float)]

        logging.info(
            f"MGA oracle: projection model added (n_z = {self.n_z} axes on "
            f"dim {Z_DIM!r}, include_cost = {self.include_cost})"
        )

    def find_nearest_point(self, trial_point: np.ndarray):
        """Callback for pyoNearOpt ORACLE.

        trial_point arrives in ORACLE polytope coordinates (always normalised;
        augmented with a cost coordinate as the last entry when include_cost).
        Returns (z_feas, dist, mu_cut, b_cut, flag) in the SAME coordinates;
        mu_cut is L2-normalised and b_cut = mu_cut @ z_feas, possibly relaxed
        by the cut-validity guard so no known inner point is cut off (see
        CUT_GUARD_TRIGGER).

        Coordinate algebra:
            trial_raw = offset + trial_norm * U_tilde         (proj-eq RHS)
            dist      = t*  — the projection LP minimises the NORMALISED
                        L-inf distance directly (scaled t-constraints), so no
                        post-hoc recomputation is needed or correct.
            mu_norm   = mu_raw o U_tilde   (mu_raw = dual of the raw proj eq)
            b_cut     = mu_norm @ z_feas_norm
        The affine cost offset C* cancels in the cut, so only U_tilde scales mu.
        """
        if trial_point.shape != (self.n_explore,):
            raise ValueError(
                f"find_nearest_point: expected shape ({self.n_explore},), "
                f"got {trial_point.shape}"
            )

        # --- Design axes: trial point -> raw coordinates, set proj-eq RHS ---
        # trial_design_raw = trial_norm * U_g* (the design offset is 0).
        trial_design_raw = trial_point[:self.n_z] * self.u_star
        for i in range(self.n_z):
            self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].rhs = float(
                trial_design_raw[i]
            )

        # --- Cost axis: trial point -> raw cost, set cost proj-eq RHS ---
        if self.include_cost:
            trial_c = float(trial_point[self.n_z])
            trial_c_raw = self.c_star + trial_c * self.epsilon * self.c_star
            trial_c_da = xr.DataArray(
                np.array([trial_c_raw]),
                dims="mga_oracle_scalar_dim",
                coords={"mga_oracle_scalar_dim": [0]},
            )
            self.model.constraints["mga_oracle_proj_eq_cost"].rhs = trial_c_da

        # Objective: min t. Sum collapses the trivial mga_oracle_scalar_dim
        # back to a scalar expression for the objective.
        self.model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self._iter_count}"
        logging.info(
            f"MGA oracle: starting iteration {self._iter_count}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        try:
            self._solve_and_postprocess(label)
        except RuntimeError:
            # One numerically distressed projection (e.g. barrier iteration
            # limit — run-14 died this way at iter 189) must not kill the
            # run. Retry once (threaded barrier runs are not deterministic);
            # if it fails again, return a zero-distance copy of the previous
            # inner point: ORACLE's identical-point check then ends the loop
            # GRACEFULLY, so the polytope artifacts and the final deep
            # certificate are still produced.
            logging.exception(
                f"MGA oracle iter {self._iter_count}: projection solve "
                f"failed; retrying once."
            )
            try:
                self._solve_and_postprocess(label)
            except RuntimeError:
                logging.error(
                    "MGA oracle: projection failed twice; returning a "
                    "sentinel point so ORACLE stops gracefully (artifacts + "
                    "final certificate still run)."
                )
                z_prev = np.asarray(self._inner_points[-1], dtype=float).copy()
                self._iter_count += 1
                return z_prev, 0.0, None, None, 0

        # z_feas in polytope coordinates (already normalised/augmented).
        z_feas = self._extract_z()

        # dist = t* = the normalised L-inf projection distance.
        dist = float(self.t_var.solution.values[0])

        # --- Cutting hyperplane: dual of the projection equality, rescaled ---
        # mu_raw = dual of the raw proj eq (sign convention matches scipy
        # linprog eqlin.marginals). mu_norm = mu_raw o U_g*.
        mu_design_raw = np.array(
            [
                float(
                    self.model.constraints[f"mga_oracle_proj_eq_axis{i}"]
                    .dual.values
                )
                for i in range(self.n_z)
            ],
            dtype=float,
        )
        mu_cut = mu_design_raw * self.u_star

        if self.include_cost:
            mu_c_raw = float(
                self.model.constraints["mga_oracle_proj_eq_cost"].dual.values[0]
            )
            mu_c = mu_c_raw * self.epsilon * self.c_star
            mu_cut = np.append(mu_cut, mu_c)

        # L2-normalise the (already coordinate-scaled) cut normal, then take
        # b_cut from the SAME normalised mu_cut so the pair stays consistent.
        scale = np.linalg.norm(mu_cut, ord=2)
        if scale > 1e-4:
            mu_cut = mu_cut / scale
        b_cut = float(mu_cut @ z_feas)

        # --- Cut-validity guard: never cut off a known near-optimal point ---
        # Every previous z_feas (and z*) lies in R_eps, so a valid supporting
        # hyperplane must keep them all inside. If the plane violates that
        # (bad projection duals), keep its direction but push it out to the
        # farthest known inner point. See CUT_GUARD_TRIGGER.
        if self._inner_points is not None:
            pts = np.vstack(self._inner_points)
            worst = float((pts @ mu_cut - b_cut).max())
            if worst > CUT_GUARD_TRIGGER:
                logging.warning(
                    f"MGA oracle iter {self._iter_count}: cut would remove "
                    f"known near-optimal point(s) by up to {worst:.4g} "
                    f"(inexact projection duals?); relaxing b_cut "
                    f"{b_cut:.6g} -> {b_cut + worst:.6g}"
                )
                b_cut = float(b_cut + worst + 1e-9)
                if float(mu_cut @ trial_point) <= b_cut:
                    # Correctness first: O must stay an outer approximation.
                    # But the relaxed plane no longer excludes the trial
                    # point, so ORACLE may re-find the same trial next
                    # iteration and stop on its identical-feasible-point
                    # check — a loud failure instead of a corrupted O.
                    logging.error(
                        "MGA oracle: relaxed cut no longer excludes the "
                        "trial point — the projection duals are unreliable "
                        "this iteration (consider re-solving with "
                        "crossover). ORACLE may stall on a repeating trial "
                        "point."
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
# Event handler (mode dispatcher) and per-mode runners
# ----------------------------------------------------------------------

@EventPublisher.register(Event.after_solve)
def run_mga(*, optimization_setup, scenarios, subfolder, model_name,
            scenario_name, param_map):
    """Entry point invoked by EventPublisher after the baseline solve.

    Returns the oracle summary directory (oracle mode) or None.
    """
    mode = config.get("mode", "weights")
    epsilon = config["epsilon"]
    exclude_techs = config.get("exclude_techs", [])
    include_techs = config.get("include_techs", [])
    include_carrier_imports = config.get("include_carrier_imports", [])

    # Oracle options (ORACLE always normalises by the per-axis fmax maxima).
    ora_cfg = config.get("oracle", {})
    include_cost = bool(ora_cfg.get("include_cost", False))

    postprocess_ctx = {
        "scenarios": scenarios,
        "subfolder": subfolder,
        "model_name": model_name,
        "scenario_name": scenario_name,
        "param_map": param_map,
    }
    logging.info(
        f"MGA plugin: mode = {mode!r}, epsilon = {epsilon}, "
        f"n_exclude = {len(exclude_techs)}, n_include = {len(include_techs)}, "
        f"n_carrier_imports = {len(include_carrier_imports)}"
    )

    mga = MGA(
        optimization_setup, epsilon, postprocess_ctx,
        exclude_techs=exclude_techs,
        include_techs=include_techs,
        include_carrier_imports=include_carrier_imports,
        include_cost=include_cost,
    )
    mga.setup()

    result = None
    if mode == "weights":
        iterations = config.get("iterations", [])
        if not iterations:
            logging.warning("MGA plugin: weights mode but no iterations configured; skipping.")
            return None
        _run_weights_mode(mga, iterations)
    elif mode == "oracle":
        result = _run_oracle_mode(mga, ora_cfg, optimization_setup, postprocess_ctx)
    else:
        raise ValueError(
            f"Unknown MGA mode: {mode!r}. Expected 'weights' or 'oracle'."
        )

    logging.info("MGA plugin: complete.")
    return result


def _run_weights_mode(mga, iterations):
    """Run the configured weights-mode iterations."""
    for i, iteration in enumerate(iterations):
        if "weights" not in iteration:
            raise KeyError(f"MGA iteration {i} missing 'weights' key")
        weights = iteration["weights"]
        if not isinstance(weights, dict):
            raise TypeError(
                f"MGA iteration {i}: 'weights' must be a dict, "
                f"got {type(weights).__name__}"
            )
        mga.run_iteration(weights, i)


def _run_oracle_mode(mga, ora_cfg, optimization_setup, postprocess_ctx):
    """Run ORACLE: fmax LPs, projection model, refinement loop, artifacts."""
    from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm
    from pyoNearOpt.polytope_approximation.approximation_class import approximation
    import pyomo.environ as pyo

    max_iter = ora_cfg.get("max_iterations", 200)
    if "tolerance" not in ora_cfg:
        raise ValueError("MGA oracle: 'tolerance' is required.")
    tol = float(ora_cfg["tolerance"])
    Md_override = ora_cfg.get("Md_override", 1e8)
    t_max_override = ora_cfg.get("t_max_override", None)
    use_bigM = bool(ora_cfg.get("use_bigM", False))
    formulation = str(ora_cfg.get("formulation", "kkt_milp")).lower()
    if formulation not in ("kkt_milp", "dual_bilinear"):
        raise ValueError(
            f"MGA oracle: unknown formulation {formulation!r}; expected "
            f"'kkt_milp' or 'dual_bilinear'."
        )
    if formulation == "dual_bilinear" and use_bigM:
        logging.warning(
            "MGA oracle: use_bigM=true is ignored with "
            "formulation='dual_bilinear' (no complementarity constraints)."
        )

    # fmax: solve the n_z auxiliary U_i* LPs (always solved, no cache; each
    # full solution is saved via Postprocess) BEFORE the projection model is
    # added and before the ORACLE loop starts. These supply both the
    # normalisation denominators and the initial outer box.
    mga.compute_fmax_normalization()

    # Add projection model to the linopy problem (ONCE).
    mga.setup_projection_model()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Coordinates across variables not equal.*",
            category=UserWarning,
        )
        A0, b0 = mga.build_initial_outer_approximation()
        z_star = mga.z_star_explore
        logging.info(f"MGA oracle: tol = {tol:.3g}")

        name_list = list(mga.z_names) + (
            ["net_present_cost"] if mga.include_cost else []
        )
        poly = approximation(
            A=A0, X=z_star.reshape(1, -1), b=b0,
            name_list=name_list,
            use_bigM=use_bigM,
            formulation=formulation,
        )
        logging.info(f"MGA oracle: step-2 formulation = {formulation!r}")
        # Big-M defaults are tuned for normalised toys; loosen for Crystal Ball.
        poly.Md = float(Md_override)
        if t_max_override is not None:
            poly.t_max = float(t_max_override)

    # Pyomo solver for ORACLE's internal polytope MILPs.
    # solver_io="python":  use Gurobi's Python API directly (no LP file IO).
    # manage_env=True:     create+release Gurobi env in the constructor;
    #                      protects single-use academic licenses from hanging.
    # Access-time .get: the plugin loader's shallow merge replaces the whole
    # "oracle" dict, so defaults cannot live in the module-level config.
    milp_options = dict(ora_cfg.get("milp_options", DEFAULT_MILP_OPTIONS))
    if formulation == "dual_bilinear":
        # The dual reformulation's bilinear terms need Gurobi's global
        # nonconvex-QP mode; explicit so behaviour is version-independent.
        milp_options.setdefault("NonConvex", 2)
    # Gurobi's console output reaches the run log twice: once as console text
    # (LogToConsole, forced on by pyomo's tee) and once mirrored through the
    # python 'gurobipy' logger, which zen-garden's bare-format stdout logging
    # prints identically. Silence the logger mirror; keep the console copy.
    logging.getLogger("gurobipy").propagate = False

    pyomo_solver = pyo.SolverFactory(
        "gurobi", solver_io="python", manage_env=True
    )
    pyomo_solver.set_options(
        " ".join(f"{k}={v}" for k, v in milp_options.items())
    )

    algo = ORACLEAlgorithm(
        poly_approx=poly,
        find_nearest_point=mga.find_nearest_point,
        max_iterations=max_iter,
        tol=tol,
        pyomo_solver=pyomo_solver,
        print_lv=1,
    )

    # Summary folder lands inside folder_output, as a sibling of the
    # per-iteration Postprocess folders. postprocess_ctx["subfolder"] is
    # only a relative sub-path (empty for non-scenario runs), so using it
    # alone would resolve against the process cwd instead.
    out = (
        Path(optimization_setup.analysis.folder_output)
        / f"{postprocess_ctx['model_name']}_oracle_summary"
    )
    out.mkdir(parents=True, exist_ok=True)
    df = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Coordinates across variables not equal.*",
                category=UserWarning,
            )
            df = algo.refine_approximations()
        df = _maybe_deep_certificate(df, poly, tol, ora_cfg, milp_options)
    finally:
        # Persist artifacts even if refine_approximations raised mid-way.
        _save_polytope_artifacts(mga, poly, df, tol, out)
    return out


def _maybe_deep_certificate(df, poly, tol, ora_cfg, milp_options):
    """One long, capped step-2 solve after a non-converged loop.

    Controlled by oracle.final_certificate_time_limit (seconds; 0/absent =
    off). Rationale: per-iteration solves get a short TimeLimit because the
    refinement loop only needs the witness, while the PROOF (the metric)
    is best bought once, on the final geometry. This runs that one deep
    solve on the server right after the loop, using the loop's last
    reported metric as a proven-valid cap (the ratchet guarantees it is an
    upper bound of the final geometry's distance), and appends the improved
    certified value as an extra row so the saved npz carries the best-known
    final_max_min_distance / converged flag. Identical mathematics to a
    post-hoc run of the diagnostics certifier on the saved polytope: the
    metric depends only on (A, b, X).
    """
    import pandas as pd
    import pyomo.environ as pyo

    final_tl = float(ora_cfg.get("final_certificate_time_limit", 0) or 0)
    if df is None or len(df) == 0 or final_tl <= 0:
        return df
    last = float(df["max_min_distance"].iloc[-1])
    if last <= tol:
        return df

    logging.info(
        f"MGA oracle: loop ended at {last:.4g} > tol {tol:.4g}; running the "
        f"final deep certificate (TimeLimit = {final_tl:.0f}s, cap = {last:.4g})."
    )
    try:
        poly.t_max = last  # the run's own reported metric: a proven cap
        poly.inner_outer_model()
        solver = pyo.SolverFactory("gurobi", solver_io="python", manage_env=True)
        opts = dict(milp_options)
        opts["TimeLimit"] = final_tl
        solver.set_options(" ".join(f"{k}={v}" for k, v in opts.items()))
        # load_solutions=False: only the dual bound is needed, and a
        # solution-less timeout must not raise.
        solver.solve(poly.out_inner, load_solutions=False, tee=True)
        deep = min(last, -float(solver._solver_model.ObjBound))
    except Exception:
        logging.exception(
            "MGA oracle: final deep certificate failed; keeping the loop's metric."
        )
        return df

    logging.info(
        f"MGA oracle: final deep certificate {last:.4g} -> {deep:.4g} "
        f"({'<= tol: CONVERGED (certified)' if deep <= tol else '> tol: not converged'})."
    )
    row = {c: None for c in df.columns}
    row["max_min_distance"] = deep
    if "iteration" in df.columns:
        row["iteration"] = int(df["iteration"].iloc[-1]) + 1
    if "max_min_solve_time" in df.columns:
        row["max_min_solve_time"] = final_tl
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def _save_polytope_artifacts(mga, poly, df, tol, out):
    """Persist polytope npz + diagnostics csv and log the convergence outcome.

    Runs in _run_oracle_mode's `finally` so a mid-loop exception (Big-M
    violation, infeasible projection, ...) does not lose the completed
    iterations; df is None in that case.
    """
    if df is not None:
        final_dist = float(df["max_min_distance"].iloc[-1])
        n_iters_done = len(df)
        converged = bool(final_dist <= tol)
    else:
        final_dist = float("nan")
        n_iters_done = 0
        converged = False

    # Self-sufficient polytope file (schema owned by polytope_io): everything
    # needed to de-normalise/interpret it. u_tilde/offset are NOT stored
    # (exact repackaging of u_star, c_star, epsilon — see MGA.polytope_metadata).
    meta = mga.polytope_metadata()
    # Original physical unit strings, aligned 1:1 with name_list (design axes
    # then the cost axis); "" where a unit is unavailable.
    unit_by_name = {a["name"]: (a["unit"] or "") for a in meta["axes"]}
    if meta["cost_axis"]:
        unit_by_name[meta["cost_axis"]] = meta["cost_unit"] or ""
    # Name the polytope file after the run: the last "_"-token of the
    # output folder (e.g. ".../cb_2050gf_ORACLE_06" -> "polytope_06.npz"),
    # so the file is self-identifying when collected across runs.
    run_id = Path(mga.optimization_setup.analysis.folder_output).name.split("_")[-1]
    poly_file = f"polytope_{run_id}.npz" if run_id else "polytope.npz"
    save_polytope(out / poly_file, Polytope(
        A=poly.A, b=poly.b, X=poly.X,
        names=[str(n) for n in poly.name_list],
        u_star=mga.u_star,
        c_star=float(mga.c_star),
        epsilon=float(mga.epsilon),
        cost_axis=meta["cost_axis"] or "",
        z_star=mga._z_star_design_raw,
        units=[unit_by_name.get(str(n), "") for n in poly.name_list],
        tolerance=float(tol),
        converged=bool(converged),
        final_max_min_distance=float(final_dist),
        meta=meta,
    ))
    if df is not None:
        df.to_csv(out / "diagnostics.csv", index=False)
        if converged:
            logging.info(
                f"MGA oracle: CONVERGED in {n_iters_done} iterations. "
                f"Final max_min_distance = {final_dist:.4g} <= tol = {tol:.4g}."
            )
        else:
            logging.warning(
                f"MGA oracle: did NOT converge after {n_iters_done} iterations. "
                f"Final max_min_distance = {final_dist:.4g}, tol = {tol:.4g}. "
                f"Increase max_iterations or relax tolerance."
            )
    else:
        logging.warning(
            "MGA oracle: refine_approximations did not return a result "
            "(likely raised mid-iteration). See traceback above."
        )
    logging.info(f"MGA oracle: artifacts saved to {out} (polytope: {poly_file})")