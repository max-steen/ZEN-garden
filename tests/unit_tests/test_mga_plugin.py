"""Fast unit tests for the MGA plugin's model-independent parts.

These run in milliseconds and need neither a solver nor a dataset: axis
config parsing, the config-key validation, row normalisation, and the
polytope npz round-trip. The end-to-end behaviour is covered by the gated
smoke test in tests/test_mga_oracle_smoke.py.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pint
import pytest

from zen_garden.plugins.mga.axes import Axis, axis_physical_unit, build_axis_groups
from zen_garden.plugins.mga.plugin import MGA, normalise_rows, validate_config
from zen_garden.plugins.mga.polytope_io import (
    CARRIER_IMPORT,
    TECH_CAPACITY,
    TOTAL_COST,
    Polytope,
    load_polytope,
    norm_to_phys,
    phys_to_norm,
    save_polytope,
)

TECHS = ["nuclear", "pv", "battery", "hydro_a", "hydro_b"]
CARRIERS = ["biomass", "hydrogen"]


# ---------------------------------------------------------------- axis config


def test_singleton_and_lumped_axes_keep_user_order():
    tech_groups, carrier_groups = build_axis_groups(
        ["nuclear", {"hydro": ["hydro_a", "hydro_b"]}],
        ["biomass"],
        TECHS,
        CARRIERS,
    )
    assert tech_groups == [("nuclear", ["nuclear"]), ("hydro", ["hydro_a", "hydro_b"])]
    assert carrier_groups == [("biomass", ["biomass"])]


def test_empty_config_yields_no_axes():
    assert build_axis_groups(None, None, TECHS, CARRIERS) == ([], [])


@pytest.mark.parametrize(
    "technologies, carriers",
    [
        (["typo"], None),  # unknown technology
        (None, ["typo"]),  # unknown carrier
        (["nuclear", "nuclear"], None),  # duplicate axis name
        ([{"nuclear": ["pv"]}], None),  # group shadows a tech
        ([{"g": ["nuclear"]}, {"h": ["nuclear"]}], None),  # member twice
        ([{"g": ["nuclear"], "h": ["pv"]}], None),  # two-key dict
        ([{"g": []}], None),  # empty member list
        ([42], None),  # not a name or dict
    ],
)
def test_invalid_axis_configs_are_rejected(technologies, carriers):
    with pytest.raises(ValueError):
        build_axis_groups(technologies, carriers, TECHS, CARRIERS)


def test_axis_name_cannot_be_used_twice_across_blocks():
    with pytest.raises(ValueError):
        build_axis_groups(
            [{"shared": ["nuclear"]}], [{"shared": ["biomass"]}], TECHS, CARRIERS
        )


# ------------------------------------------------------------ config validation


def test_valid_config_passes():
    validate_config(
        {
            "epsilon": 0.1,
            "mode": "oracle",
            "axes": {"technologies": ["nuclear"], "include_cost": True},
            "oracle": {"tolerance": 0.1, "max_min": {"use_bigM": True}},
        }
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {"epsilonn": 0.1},  # top-level typo
        {"axes": {"technolgies": []}},  # axes typo
        {"oracle": {"tolerance": 0.1, "max_iter": 10}},  # oracle typo
        {"oracle": {"max_min": {"milp_options": {}}}},  # renamed key
    ],
)
def test_unknown_config_keys_are_rejected(cfg):
    with pytest.raises(ValueError, match="Unknown MGA config key"):
        validate_config(cfg)


# -------------------------------------------------------------- supplied bounds


def _mga_with_design_axes(*names):
    """A bare MGA stand-in with only design_axes, enough for bounds parsing."""
    axes = [Axis(n, TECH_CAPACITY, (n,), "power") for n in names]
    return SimpleNamespace(design_axes=axes)


def test_supplied_bounds_are_read_in_axis_order():
    mga = _mga_with_design_axes("nuclear", "pv")
    lower, upper = MGA._read_supplied_bounds(
        mga, {"pv": [1.0, 9000.0], "nuclear": [0.0, 110.0]}
    )
    assert lower.tolist() == [0.0, 1.0]
    assert upper.tolist() == [110.0, 9000.0]


@pytest.mark.parametrize(
    "bounds",
    [
        {},  # missing every axis
        {"nuclear": [0.0, 110.0]},  # missing pv
        {"nuclear": [0.0, 110.0], "pv": [0.0, 1.0], "typo": [0.0, 1.0]},
        "vmm-typo",  # not a dict
    ],
)
def test_supplied_bounds_must_cover_design_axes_exactly(bounds):
    mga = _mga_with_design_axes("nuclear", "pv")
    with pytest.raises(ValueError):
        MGA._read_supplied_bounds(mga, bounds)


# ----------------------------------------------------------- row normalisation


def test_normalise_rows_gives_unit_rows_and_keeps_the_half_spaces():
    A = np.array([[3.0, 0.0], [0.0, 743303.0], [-3.0, 4.0]])
    b = np.array([3.0, 743303.0, 5.0])
    point = np.array([0.5, 0.25])
    An, bn = normalise_rows(A, b)

    assert np.allclose(np.linalg.norm(An, axis=1), 1.0)
    # membership is unchanged: same sign of the residual, row by row
    assert np.array_equal(A @ point <= b, An @ point <= bn)
    assert np.allclose(bn[:2], 1.0)  # z <= 1 in both rows


# ------------------------------------------------------------- coordinate maps


def test_norm_phys_round_trip():
    scale = np.array([110.0, 9834.0, 3.0e8])
    offset = np.array([0.0, 0.0, 1.25e9])
    norm = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]])
    assert np.allclose(
        phys_to_norm(norm_to_phys(norm, scale, offset), scale, offset), norm
    )


def test_cost_axis_maps_optimum_to_zero_and_budget_to_one():
    c_star, epsilon = 1000.0, 0.1
    scale, offset = np.array([epsilon * c_star]), np.array([c_star])
    assert norm_to_phys(np.array([0.0]), scale, offset)[0] == c_star
    assert norm_to_phys(np.array([1.0]), scale, offset)[0] == 1100.0


def test_coordinate_map_rejects_wrong_axis_count():
    with pytest.raises(ValueError):
        norm_to_phys(np.zeros(2), np.ones(3), np.zeros(3))


# ------------------------------------------------------------------ npz schema


def _polytope() -> Polytope:
    return Polytope(
        A=np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        b=np.array([0.0, 1.0, 1.0, 1.0]),
        X=np.array([[0.1, 0.2, 0.0], [1.0, 0.5, 0.4]]),
        names=["nuclear", "biomass", "net_present_cost"],
        kinds=[TECH_CAPACITY, CARRIER_IMPORT, TOTAL_COST],
        units=["gigawatt", "gigawatt_hour", "megaEuro"],
        scale=np.array([110.0, 777923.0, 1.25e8]),
        offset=np.array([0.0, 0.0, 1.25e9]),
        bounds_phys=np.array([[0.0, 110.0], [0.0, 777923.0], [1.25e9, 1.3750e9]]),
        z_star_phys=np.array([11.0, 155584.6, 1.25e9]),
        c_star=1.25e9,
        epsilon=0.1,
        tolerance=0.1,
        converged=False,
        final_max_min_distance=0.26,
        n_initial_rows=4,
        point_origin=["z_star", "max:nuclear"],
        meta={"axes": [{"name": "nuclear"}], "normalisation": "affine"},
        run={"formulation": "dual_bilinear", "iterations_done": 7},
    )


def test_polytope_npz_round_trip(tmp_path):
    original = _polytope()
    path = tmp_path / "polytope_test.npz"
    save_polytope(path, original)
    loaded = load_polytope(path)

    for field in ("A", "b", "X", "scale", "offset", "bounds_phys", "z_star_phys"):
        assert np.allclose(getattr(loaded, field), getattr(original, field))
    assert loaded.names == original.names
    assert loaded.kinds == original.kinds
    assert loaded.units == original.units
    assert loaded.point_origin == original.point_origin
    assert loaded.meta == original.meta
    assert loaded.run == original.run
    assert loaded.n_initial_rows == 4
    assert loaded.converged is False
    assert loaded.c_star == original.c_star


def test_loaded_polytope_exposes_axis_roles(tmp_path):
    path = tmp_path / "polytope_test.npz"
    save_polytope(path, _polytope())
    loaded = load_polytope(path)

    assert loaded.n_axes == 3
    assert loaded.cost_col == 2
    assert loaded.design_cols == [0, 1]
    assert loaded.design_names == ["nuclear", "biomass"]
    assert np.allclose(loaded.to_norm(loaded.to_phys(loaded.X)), loaded.X)


def test_loader_reports_every_missing_schema_key(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, A=np.eye(2), b=np.ones(2))
    with pytest.raises(KeyError, match="missing schema keys"):
        load_polytope(path)


# ------------------------------------------------------------- physical units


def _units_series(index_names, tuples, unit_strings):
    index = pd.MultiIndex.from_tuples(tuples, names=index_names)
    return pd.Series(unit_strings, index=index, dtype=str)


# Level names as ZEN-garden builds them for the unit series: the dimensions'
# documentation names, not the set names the variables are indexed by.
CAPACITY_UNITS = _units_series(
    ["technology", "capacity_type", "location", "year"],
    [
        ("nuclear", "power", "DE", 2050),
        ("battery", "power", "DE", 2050),
        ("battery", "energy", "DE", 2050),
    ],
    ["gigawatt", "gigawatt", "gigawatt_hour"],
)
IMPORT_UNITS = _units_series(
    ["carrier", "node", "time_operation"],
    [("biomass", "DE", 0), ("hydrogen", "DE", 0)],
    ["gigawatt", "gigawatt"],
)


def test_tech_axis_unit_follows_the_selected_capacity_type():
    units = {"capacity_addition": CAPACITY_UNITS}
    power = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
    energy = Axis("battery", TECH_CAPACITY, ("battery",), "energy")

    assert axis_physical_unit(power, units, pint.UnitRegistry()) == "gigawatt"
    assert axis_physical_unit(energy, units, pint.UnitRegistry()) == "gigawatt_hour"


def test_carrier_axis_unit_is_annualised():
    axis = Axis("biomass", CARRIER_IMPORT, ("biomass",), None)
    unit = axis_physical_unit(axis, {"flow_import": IMPORT_UNITS}, pint.UnitRegistry())
    assert unit == "gigawatt * hour"


def test_cost_axis_reads_the_cost_variable_unit():
    axis = Axis("net_present_cost", TOTAL_COST, (), None)
    units = {"net_present_cost": pd.Series(["megaEuro"], dtype=str)}
    assert axis_physical_unit(axis, units, pint.UnitRegistry()) == "megaEuro"


def test_unit_is_none_when_the_model_tracks_no_units():
    axis = Axis("nuclear", TECH_CAPACITY, ("nuclear",), "power")
    assert axis_physical_unit(axis, {}, pint.UnitRegistry()) is None
