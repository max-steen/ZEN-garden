"""Exploration-axis definitions for the MGA plugin.

An axis is one coordinate of the explored near-optimal space: the summed
capacity addition of a group of technologies, the duration-weighted annual
import of a group of carriers, or the total system cost. This module owns
everything about axes that does not need the optimization model: the Axis
type, the parsing and validation of the axis config lists, and the
physical-unit lookup for the polytope metadata. Model-coupled axis logic
lives in plugin.MGA.
"""

from dataclasses import dataclass

import numpy as np

from .polytope_io import CARRIER_IMPORT, TECH_CAPACITY, TOTAL_COST


@dataclass(frozen=True)
class Axis:
    """One exploration axis of the MGA polytope.

    TECH_CAPACITY axes sum capacity_addition over the member technologies,
    restricted to the selected capacity type; CARRIER_IMPORT axes sum the
    duration-weighted annual flow_import over the member carriers; the single
    TOTAL_COST axis is the model's net present cost and has no members.
    capacity_type is the "+"-joined selected type(s) for tech axes and None
    otherwise.
    """

    name: str
    kind: str
    members: tuple[str, ...]
    capacity_type: str | None


def build_axis_groups(technologies, carrier_imports, all_technologies,
                      all_carriers):
    """Turn the axis config lists into ordered (name, members) groups.

    Returns (tech_groups, carrier_groups), each in the user's order.
    """
    tech_set, carrier_set = set(all_technologies), set(all_carriers)
    tech_groups = _parse_axis_list(
        technologies, tech_set, tech_set, "axes.technologies"
    )
    # Axis names share one namespace with the model names in the polytope
    # file, so carrier groups must not reuse a technology or carrier name.
    carrier_groups = _parse_axis_list(
        carrier_imports, carrier_set, tech_set | carrier_set,
        "axes.carrier_imports",
    )
    duplicates = {n for n, _ in tech_groups} & {n for n, _ in carrier_groups}
    if duplicates:
        raise ValueError(
            f"MGA: axis name(s) {sorted(duplicates)} used for both a "
            f"technology and a carrier axis."
        )
    return tech_groups, carrier_groups


def _parse_axis_list(entries, valid_members, reserved_names, label):
    """Parse one axis config list into ordered (name, members) tuples.

    Each entry is an axis name (singleton axis) or a single-key dict
    ``{group_name: [member, ...]}`` (lumped axis). Axis names must be unique,
    group names must not shadow an existing model name, and each member may
    appear in at most one axis (it would otherwise be counted twice).
    """
    groups = []
    seen_names = set()
    axis_of_member = {}
    unknown = []
    for entry in entries or []:
        if isinstance(entry, str):
            name, members = entry, [entry]
        elif isinstance(entry, dict) and len(entry) == 1:
            name, members = next(iter(entry.items()))
            if name in reserved_names:
                raise ValueError(f"MGA {label}: group name {name!r} shadows an "
                                 f"existing technology or carrier name.")
        else:
            raise ValueError(f"MGA {label}: invalid entry {entry!r}, expected "
                             f"a name or a single {{group: [members]}} dict.")
        well_formed = (isinstance(name, str) and name
                       and isinstance(members, list) and members
                       and all(isinstance(m, str) and m for m in members))
        if not well_formed:
            raise ValueError(f"MGA {label}: invalid entry {entry!r}.")
        if name in seen_names:
            raise ValueError(f"MGA {label}: duplicate axis name {name!r}.")
        seen_names.add(name)
        for member in members:
            if member not in valid_members:
                unknown.append(member)
            elif member in axis_of_member:
                raise ValueError(f"MGA {label}: {member!r} appears in both axis "
                                 f"{axis_of_member[member]!r} and {name!r}.")
            else:
                axis_of_member[member] = name
        groups.append((name, list(members)))
    if unknown:
        raise KeyError(f"MGA {label}: unknown names {sorted(set(unknown))}")
    return groups


# Substring needles for locating levels in the units-series MultiIndex; the
# level names vary across ZEN-garden versions (e.g. "technology" vs
# "set_technologies").
_LEVEL_TECH = "technolog"
_LEVEL_CAPACITY_TYPE = "capacity_type"
_LEVEL_CARRIER = "carrier"


def _find_level(index, needle):
    """Name of the first index level containing `needle`, or None."""
    return next((lvl for lvl in index.names if needle in lvl), None)


def axis_physical_unit(axis, units, ureg, cost_variable="net_present_cost"):
    """Physical unit string of one axis value, or None if unavailable.

    Tech axes read the capacity_addition unit at the selected capacity type;
    carrier axes annualise the instantaneous flow_import unit (x hour); the
    cost axis reads the cost variable's unit. Heterogeneous lumps yield a
    ' + '-joined string. `units` is the model's variable-unit mapping, which
    is empty when unit tracking is switched off.
    """
    members = list(axis.members)
    if axis.kind == TOTAL_COST:
        series = units.get(cost_variable)
        if series is None:
            return None
        found = sorted({str(u) for u in np.atleast_1d(np.asarray(series))})
        return " + ".join(found) if found else None

    if axis.kind == TECH_CAPACITY:
        series = units.get("capacity_addition")
        if series is None:
            return None
        tech_level = _find_level(series.index, _LEVEL_TECH)
        type_level = _find_level(series.index, _LEVEL_CAPACITY_TYPE)
        if tech_level is None or type_level is None:
            return None
        mask = (series.index.get_level_values(tech_level).isin(members)
                & series.index.get_level_values(type_level)
                .isin(axis.capacity_type.split("+")))
        found = sorted({str(u) for u in series[mask].to_numpy()})
        return " + ".join(found) if found else None

    if axis.kind != CARRIER_IMPORT:
        return None
    series = units.get("flow_import")
    if series is None:
        return None
    carrier_level = _find_level(series.index, _LEVEL_CARRIER)
    if carrier_level is None:
        return None
    mask = series.index.get_level_values(carrier_level).isin(members)
    annual = set()
    for unit in {str(u) for u in series[mask].to_numpy()}:
        try:
            annual.add(str(ureg(f"({unit}) * hour").units))
        except Exception:
            annual.add(f"({unit}) * hour")
    return " + ".join(sorted(annual)) if annual else None
