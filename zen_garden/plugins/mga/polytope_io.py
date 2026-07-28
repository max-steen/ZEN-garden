"""Reader/writer and de-normalisation helpers for MGA ORACLE polytope files.

This module is the single owner of the polytope npz schema written by the MGA
plugin and read by its consumers. It is deliberately dependency-light (numpy +
stdlib only) and side-effect-free: importing it does NOT import the MGA plugin
and therefore registers no event handlers.

Every exploratory variable is an axis -- a technology-capacity group, a
carrier-import group, or the total cost -- so all per-axis arrays share one
length and one order, matching the polytope's columns.

Schema:

    A               (n_rows, n_axes)     outer approximation, normalised coords
    b               (n_rows,)
    X               (n_points, n_axes)   certified near-optimal points
    name_list       (n_axes,) str        axis names, in column order
    kinds           (n_axes,) str        tech_capacity | carrier_import | total_cost
    units           (n_axes,) str        physical unit, "" when unknown
    scale           (n_axes,)            physical = normalised * scale + offset
    offset          (n_axes,)
    bounds_phys     (n_axes, 2)          [lower, upper] per axis, physical units
    z_star_phys     (n_axes,)            cost-optimal baseline point, physical
    c_star          ()                   baseline net present cost
    epsilon         ()                   near-optimality slack
    tolerance       ()                   ORACLE convergence tolerance
    converged       () bool
    final_max_min_distance ()            NaN if the run raised before a result
    n_initial_rows  ()                   leading rows of A that form the
                                         initial box; the rest are cutting planes
    point_origin    (n_points,) str      z_star | max:<axis> | min:<axis> | iterate
    axis_meta_json  () str               per-axis members and capacity type
    run_json        () str               how the run was configured and ended

Normalisation is affine and per axis: physical = normalised * scale + offset.
Design axes use scale = upper bound and offset = 0, so the axis reaches 1 at
its near-optimal maximum. The cost axis uses scale = epsilon * c_star and
offset = c_star, so 0 is the cost optimum and 1 the near-optimality budget.
Both conventions are stored explicitly because they do not follow one rule.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Axis kinds, stored per axis in `kinds` and in axis_meta_json.
TECH_CAPACITY = "tech_capacity"
CARRIER_IMPORT = "carrier_import"
TOTAL_COST = "total_cost"

# Every key the schema defines; save_polytope writes all of them.
NPZ_KEYS = (
    "A",
    "b",
    "X",
    "name_list",
    "kinds",
    "units",
    "scale",
    "offset",
    "bounds_phys",
    "z_star_phys",
    "c_star",
    "epsilon",
    "tolerance",
    "converged",
    "final_max_min_distance",
    "n_initial_rows",
    "point_origin",
    "axis_meta_json",
    "run_json",
)


@dataclass(frozen=True)
class Polytope:
    """A loaded (or to-be-written) ORACLE polytope with its normalisation."""

    A: np.ndarray
    b: np.ndarray
    X: np.ndarray
    names: list[str]
    kinds: list[str]
    units: list[str]
    scale: np.ndarray
    offset: np.ndarray
    bounds_phys: np.ndarray
    z_star_phys: np.ndarray
    c_star: float
    epsilon: float
    tolerance: float
    converged: bool
    final_max_min_distance: float
    n_initial_rows: int
    point_origin: list[str]
    meta: dict = field(repr=False)
    run: dict = field(repr=False)

    @property
    def n_axes(self) -> int:
        """Number of exploratory variables (columns of A and X)."""
        return len(self.names)

    @property
    def cost_col(self) -> int | None:
        """Column index of the cost axis, or None when the run had none."""
        return next((i for i, k in enumerate(self.kinds) if k == TOTAL_COST), None)

    @property
    def design_cols(self) -> list[int]:
        """Column indices of the design (non-cost) axes."""
        return [i for i, k in enumerate(self.kinds) if k != TOTAL_COST]

    @property
    def design_names(self) -> list[str]:
        return [self.names[i] for i in self.design_cols]

    @property
    def axes(self) -> list[dict]:
        """Per-axis metadata dicts (name, kind, members, capacity_type, unit)."""
        return self.meta["axes"]

    def to_phys(self, Z) -> np.ndarray:
        """Map normalised coordinates (..., n_axes) to physical units."""
        return norm_to_phys(Z, self.scale, self.offset)

    def to_norm(self, Z) -> np.ndarray:
        """Inverse of to_phys."""
        return phys_to_norm(Z, self.scale, self.offset)


def save_polytope(path, poly: Polytope) -> None:
    """Write `poly` to `path` as an npz in the schema above."""
    np.savez(
        Path(path),
        A=poly.A,
        b=poly.b,
        X=poly.X,
        name_list=np.array(poly.names),
        kinds=np.array(poly.kinds),
        units=np.array(poly.units),
        scale=np.asarray(poly.scale, dtype=float),
        offset=np.asarray(poly.offset, dtype=float),
        bounds_phys=np.asarray(poly.bounds_phys, dtype=float),
        z_star_phys=np.asarray(poly.z_star_phys, dtype=float),
        c_star=float(poly.c_star),
        epsilon=float(poly.epsilon),
        tolerance=float(poly.tolerance),
        converged=bool(poly.converged),
        final_max_min_distance=float(poly.final_max_min_distance),
        n_initial_rows=int(poly.n_initial_rows),
        point_origin=np.array(poly.point_origin),
        axis_meta_json=np.array(json.dumps(poly.meta)),
        run_json=np.array(json.dumps(poly.run)),
    )


def load_polytope(path) -> Polytope:
    """Load a polytope npz written by save_polytope.

    Raises KeyError for files that do not carry the full schema and
    ValueError for internally inconsistent shapes.
    """
    path = Path(path)
    d = np.load(path)  # the schema has no object arrays; allow_pickle stays False
    missing = [k for k in NPZ_KEYS if k not in d.files]
    if missing:
        raise KeyError(f"{path} is missing schema keys {missing}.")

    A = np.asarray(d["A"], dtype=float)
    b = np.asarray(d["b"], dtype=float).ravel()
    X = np.asarray(d["X"], dtype=float)
    names = [str(n) for n in d["name_list"]]
    kinds = [str(k) for k in d["kinds"]]
    units = [str(u) for u in d["units"]]
    scale = np.asarray(d["scale"], dtype=float).ravel()
    offset = np.asarray(d["offset"], dtype=float).ravel()
    bounds = np.asarray(d["bounds_phys"], dtype=float).reshape(-1, 2)
    z_star = np.asarray(d["z_star_phys"], dtype=float).ravel()

    n_axes = len(names)
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{path}: b has {b.shape[0]} rows, A has {A.shape[0]}")
    lengths = {
        "A columns": A.shape[1],
        "X columns": X.shape[1],
        "kinds": len(kinds),
        "units": len(units),
        "scale": len(scale),
        "offset": len(offset),
        "bounds_phys": bounds.shape[0],
        "z_star_phys": len(z_star),
    }
    disagreeing = {k: v for k, v in lengths.items() if v != n_axes}
    if disagreeing:
        raise ValueError(f"{path}: {disagreeing} disagree with {n_axes} axis names.")
    if np.any(scale == 0.0):
        raise ValueError(f"{path}: scale contains zeros; cannot de-normalise.")

    return Polytope(
        A=A,
        b=b,
        X=X,
        names=names,
        kinds=kinds,
        units=units,
        scale=scale,
        offset=offset,
        bounds_phys=bounds,
        z_star_phys=z_star,
        c_star=float(d["c_star"]),
        epsilon=float(d["epsilon"]),
        tolerance=float(d["tolerance"]),
        converged=bool(d["converged"]),
        final_max_min_distance=float(d["final_max_min_distance"]),
        n_initial_rows=int(d["n_initial_rows"]),
        point_origin=[str(o) for o in d["point_origin"]],
        meta=json.loads(str(d["axis_meta_json"])),
        run=json.loads(str(d["run_json"])),
    )


# ---------------------------------------------------------------------------
# Normalised <-> physical conversion (standalone, array-shape (..., n_axes))
# ---------------------------------------------------------------------------


def norm_to_phys(Z, scale, offset) -> np.ndarray:
    """Map normalised coordinates to physical units: Z * scale + offset."""
    Z = np.asarray(Z, dtype=float)
    scale = np.asarray(scale, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if Z.shape[-1] != scale.shape[-1]:
        raise ValueError(
            f"last dimension {Z.shape[-1]} does not match {scale.shape[-1]} axes"
        )
    return Z * scale + offset


def phys_to_norm(Z, scale, offset) -> np.ndarray:
    """Inverse of norm_to_phys: (Z - offset) / scale."""
    Z = np.asarray(Z, dtype=float)
    scale = np.asarray(scale, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if Z.shape[-1] != scale.shape[-1]:
        raise ValueError(
            f"last dimension {Z.shape[-1]} does not match {scale.shape[-1]} axes"
        )
    return (Z - offset) / scale
