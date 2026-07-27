"""Fast unit tests for the MGA plugin's model-independent parts.

These run in milliseconds and need neither a solver nor a dataset: axis
config parsing, the config-key validation, row normalisation, and the
polytope npz round-trip. The end-to-end behaviour is covered by the gated
smoke test in tests/test_mga_oracle_smoke.py.
"""

import numpy as np
import pytest

from zen_garden.plugins.mga.axes import build_axis_groups
from zen_garden.plugins.mga.plugin import normalise_rows, validate_config
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
        ["nuclear", {"hydro": ["hydro_a", "hydro_b"]}], ["biomass"],
        TECHS, CARRIERS,
    )
    assert tech_groups == [("nuclear", ["nuclear"]),
                           ("hydro", ["hydro_a", "hydro_b"])]
    assert carrier_groups == [("biomass", ["biomass"])]


def test_empty_config_yields_no_axes():
    assert build_axis_groups(None, None, TECHS, CARRIERS) == ([], [])


@pytest.mark.parametrize("technologies, carriers, expected", [
    (["typo"], None, KeyError),                       # unknown technology
    (None, ["typo"], KeyError),                       # unknown carrier
    (["nuclear", "nuclear"], None, ValueError),       # duplicate axis name
    ([{"nuclear": ["pv"]}], None, ValueError),        # group shadows a tech
    ([{"g": ["nuclear"]}, {"h": ["nuclear"]}], None, ValueError),  # member twice
    ([{"g": ["nuclear"], "h": ["pv"]}], None, ValueError),  # two-key dict
    ([{"g": []}], None, ValueError),                  # empty member list
    ([42], None, ValueError),                         # not a name or dict
])
def test_invalid_axis_configs_are_rejected(technologies, carriers, expected):
    with pytest.raises(expected):
        build_axis_groups(technologies, carriers, TECHS, CARRIERS)


def test_axis_name_cannot_be_used_twice_across_blocks():
    with pytest.raises(ValueError):
        build_axis_groups([{"shared": ["nuclear"]}], [{"shared": ["biomass"]}],
                          TECHS, CARRIERS)


# ------------------------------------------------------------ config validation

def test_valid_config_passes():
    validate_config({
        "epsilon": 0.1, "mode": "oracle",
        "axes": {"technologies": ["nuclear"], "include_cost": True},
        "oracle": {"tolerance": 0.1, "step2": {"use_bigM": True}},
    })


@pytest.mark.parametrize("cfg", [
    {"epsilonn": 0.1},                                  # top-level typo
    {"axes": {"technolgies": []}},                      # axes typo
    {"oracle": {"tolerance": 0.1, "max_iter": 10}},     # oracle typo
    {"oracle": {"step2": {"milp_options": {}}}},        # renamed key
])
def test_unknown_config_keys_are_rejected(cfg):
    with pytest.raises(ValueError, match="Unknown MGA config key"):
        validate_config(cfg)


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


def test_normalise_rows_leaves_a_zero_row_alone():
    An, bn = normalise_rows(np.zeros((1, 2)), np.array([1.0]))
    assert np.array_equal(An, np.zeros((1, 2))) and bn[0] == 1.0


# ------------------------------------------------------------- coordinate maps

def test_norm_phys_round_trip():
    scale = np.array([110.0, 9834.0, 3.0e8])
    offset = np.array([0.0, 0.0, 1.25e9])
    norm = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]])
    assert np.allclose(phys_to_norm(norm_to_phys(norm, scale, offset),
                                    scale, offset), norm)


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
        A=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]),
        b=np.array([0.0, 1.0, 1.0, 1.0]),
        X=np.array([[0.1, 0.2, 0.0], [1.0, 0.5, 0.4]]),
        names=["nuclear", "biomass", "net_present_cost"],
        kinds=[TECH_CAPACITY, CARRIER_IMPORT, TOTAL_COST],
        units=["gigawatt", "gigawatt_hour", "megaEuro"],
        scale=np.array([110.0, 777923.0, 1.25e8]),
        offset=np.array([0.0, 0.0, 1.25e9]),
        bounds_phys=np.array([[0.0, 110.0], [0.0, 777923.0], [1.25e9, 1.3750e9]]),
        z_star_phys=np.array([11.0, 155584.6, 1.25e9]),
        c_star=1.25e9, epsilon=0.1, tolerance=0.1, converged=False,
        final_max_min_distance=0.26, n_initial_rows=4,
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


def test_loader_rejects_files_without_a_schema_version(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, A=np.eye(2), b=np.ones(2))
    with pytest.raises(KeyError, match="schema"):
        load_polytope(path)
