# ZEN-garden — BSc thesis fork: MGA plugin

This fork contains the code of the Bachelor thesis *Efficient exploration of
near-optimal solutions in sustainable energy systems optimisation* (Max
Steen, ETH Zürich, 2026). Upstream ZEN-garden is unchanged; the thesis adds:

- **`zen_garden/plugins/mga/`** — a plugin that explores the near-optimal
  space of a solved model: all designs whose total cost stays within
  `(1 + epsilon) * C*` of the cost optimum `C*`. Two modes:
  - **weights** — classical MGA: one re-solve per user-provided weight
    vector over technology capacity additions.
  - **oracle** — the ORACLE algorithm
    ([Turan, Moret, Bardow 2026](https://doi.org/10.1016/j.compchemeng.2026.109630)):
    iteratively refines an inner and an outer polytope approximation of the
    near-optimal space until the max-min distance between them falls below a
    tolerance, so coverage of the entire space is certified.
- **The `after_solve` plugin event** — added to
  `zen_garden/plugin_system/events.py` and fired in `zen_garden/runner.py`
  once per solved scenario; the plugin registers on it.
- **Tests** — `tests/unit_tests/test_mga_plugin.py` (fast, no solver or
  dataset needed) and `tests/test_mga_oracle_smoke.py` (end-to-end oracle
  run; gated behind `RUN_MGA_SMOKE=1` and `MGA_SMOKE_DATA=<dataset dir>`).

The plugin code is organised as `plugin.py` (event handler and model
interface), `axes.py` (exploration-axis parsing), `oracle_driver.py`
(pyoNearOpt wiring), and `polytope_io.py` (the polytope npz schema with its
reader and writer).

## Requirements

- **Weights mode:** this fork alone.
- **Oracle mode:** Gurobi (with a license — the max-min solver is hardcoded
  to Gurobi) and the **pyoNearOpt** package (E. Turan, ETH Zürich; not
  public yet — request access and install manually, e.g.
  `pip install -e path/to/pyoNearOpt`). The plugin runs with the base
  (published) package; `formulation: "dual_bilinear"` needs a pyoNearOpt
  that includes this formulation.

## Configuration

Activate the plugin via the `plugins` block of `config.json`. Unknown keys
anywhere in the block are rejected.

| key | default | meaning |
|---|---|---|
| `epsilon` | 0.1 | near-optimality slack (0.1 = 10 % cost budget) |
| `mode` | `"weights"` | `"weights"` or `"oracle"` |
| `iterations` | — | weights mode: one `{"weights": {technology: weight}}` dict per re-solve |
| `axes.technologies` | — | technology axes: names, or `{group_name: [members]}` for lumped axes |
| `axes.carrier_imports` | — | carrier-import axes, same entry format |
| `axes.include_cost` | `false` | add the total-cost axis |
| `oracle.tolerance` | required | convergence tolerance (normalised coordinates) |
| `oracle.max_iterations` | 200 | iteration cap |
| `oracle.initial_bounds` | `"vmm"` | `"vmm"` (two LPs per design axis: certified bounds + extreme designs) or a dict `{axis: [lower, upper]}` covering every design axis |
| `oracle.max_min.formulation` | `"kkt_milp"` | single-level reformulation of the max-min distance problem; `"dual_bilinear"` needs a pyoNearOpt that includes it |
| `oracle.max_min.use_bigM` | `true` | complementarity encoding for `kkt_milp`: big-M (`true`) or SOS1 (`false`) |
| `oracle.max_min.big_M` | 1e8 | big-M bound on the KKT duals |
| `oracle.max_min.t_max` | 1e6 (pyoNearOpt) | cap on the max-min distance |
| `oracle.max_min.solver_options` | `{}` | Gurobi options for the max-min solves |
| `oracle.max_min.certificate_time_limit` | 0 (off) | seconds for one long final max-min solve after a non-converged loop, to tighten the stored metric |

Example (oracle mode):

```json
{
    "plugins": {
        "mga": {
            "mode": "oracle",
            "epsilon": 0.1,
            "axes": {
                "technologies": [
                    "nuclear",
                    {"hydro_lump": ["reservoir_hydro", "run-of-river_hydro"]}
                ],
                "carrier_imports": ["biomass"],
                "include_cost": true
            },
            "oracle": {
                "tolerance": 0.1,
                "max_min": {"solver_options": {"TimeLimit": 120}}
            }
        }
    }
}
```

## Outputs

Every solve is written as an ordinary ZEN-garden results folder next to the
baseline (`<model>` is the dataset name):

```text
<model>/                    baseline (written by ZEN-garden itself)
<model>_fmax_<axis>/        VMM maximum LP, one per design axis
<model>_fmin_<axis>/        VMM minimum LP, one per design axis
<model>_mga_iter_<i>/       weights mode: one folder per iteration
<model>_oracle_iter_<n>/    oracle mode: one folder per projection solve
                            (numbering matches diagnostics.csv)
<model>_oracle_summary/     polytope.npz + diagnostics.csv
```

`polytope.npz` holds the outer approximation, the certified inner points,
the normalisation, and per-axis metadata; the schema is documented in and
read back by `zen_garden/plugins/mga/polytope_io.py` (`load_polytope`).
`diagnostics.csv` is pyoNearOpt's per-iteration record.

## Limitations

- Rolling-horizon runs, scaled runs (`solver.use_scaling`), and non-cost
  objectives are rejected; see the module docstring of `plugin.py`.
- Config errors are reported only after the baseline solve (`after_solve`
  is the only plugin event available).

*Further development of the plugin continues in the separate
ZEN-garden-plugins repository; this fork is the frozen thesis state.*

<hr style="height: 5px; background-color: black;">

# ZEN-garden
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FZEN-universe%2FZEN-garden%2Fmain%2Fpyproject.toml)

[![GitHub Release](https://img.shields.io/github/v/release/ZEN-universe/ZEN-garden)](https://github.com/ZEN-universe/ZEN-garden/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/zen-garden)](https://pypi.org/project/zen-garden/)

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ZEN-universe/ZEN-garden/pytest_with_conda.yml)](https://github.com/ZEN-universe/ZEN-garden/actions)
[![Endpoint Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jacob-mannhardt/30d479a5b4c591a63b7b0f41abbce6a0/raw/zen_garden_coverage.json)](https://github.com/ZEN-universe/ZEN-garden/actions)
[![Read the Docs](https://img.shields.io/readthedocs/zen-garden?logo=readthedocs)](https://zen-garden.readthedocs.io/en/latest/index.html)

[![GitHub forks](https://img.shields.io/github/forks/ZEN-universe/ZEN-garden)](https://github.com/ZEN-universe/ZEN-garden/forks)

<img src="https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d6a9aca9-74b0-4a82-8295-43e6a78b8450" alt="drawing" width="200"/>



Welcome to the ZEN-garden! ZEN-garden is an optimization framework for energy transition pathways. 
It is currently used to model the electricity system, hydrogen value chains, and carbon capture, storage and utilization (CCUS) value chains. 
However, it is designed to be modular and flexible, and can be extended to model other types of energy systems, value chains or other network-based systems. 

ZEN-garden is developed by the [Reliability and Risk Engineering Laboratory](https://www.rre.ethz.ch/) at ETH Zurich.
<hr style="height: 5px; background-color: black;">

## Quick Start
To get started with ZEN-garden, you can follow the instructions in the [installation guide](https://zen-garden.readthedocs.io/en/latest/files/quick_start/installation.html).

If you want to use ZEN-garden without working on the codebase, run the following command:
```bash
pip install zen-garden
```
If you want to work on the codebase, fork and clone the repository and install the package in editable mode. More information on how to install the package in editable mode can be found in the [installation guide](https://zen-garden.readthedocs.io/en/latest/files/quick_start/installation.html).

## Documentation
Please refer to the documentation of the ZEN-garden framework [on Read-the-Docs](https://zen-garden.readthedocs.io/en/latest/). 
Additionally, example datasets are available in the `dataset_examples` folder and described in [the documentation](https://zen-garden.readthedocs.io/en/latest/files/zen_garden_in_detail/dataset_examples.html).

## News
Review recent modifications outlined in the [changelog](https://github.com/ZEN-universe/ZEN-garden/blob/main/CHANGELOG.md).

## Citing ZEN-garden
If you use ZEN-garden for research, please cite

Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco De Marco, Lukas Kunz, Lukas Schmidt-Engelbertz, Paolo Gabrielli, Giovanni Sansavini (2025).
ZEN-garden: Optimizing energy transition pathways with user-oriented data handling. https://www.sciencedirect.com/science/article/pii/S2352711025000263

and use the following BibTeX:
```
@article{ZENgarden2025,
title = {ZEN-garden: Optimizing Energy Transition Pathways with User-Oriented Data Handling},
author = {Mannhardt, Jacob and Ganter, Alissa and Burger, Johannes and De Marco, Francesco and Kunz, Lukas and {Schmidt-Engelbertz}, Lukas and Gabrielli, Paolo and Sansavini, Giovanni},
year = {2025},
journal = {SoftwareX},
volume = {29},
pages = {102059},
issn = {2352-7110},
doi = {10.1016/j.softx.2025.102059},
}
```
