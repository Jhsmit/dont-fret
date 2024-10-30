from __future__ import annotations

import itertools
import math
from functools import reduce
from operator import and_
from typing import TYPE_CHECKING, Literal, Optional, TypedDict, Union

import polars as pl

from dont_fret.models import PhotonData
from dont_fret.web.models import BurstFilterItem

if TYPE_CHECKING:
    from dont_fret.web.new_models import FRETNode


def chain_filters(filters: list[BurstFilterItem]) -> Union[pl.Expr, Literal[True]]:
    """Chain a list of `BurstFilterItem` objects into a single `pl.Expr` object."""
    f_exprs = list(itertools.chain(*(f.as_expr() for f in filters if f.active)))
    if f_exprs:
        return reduce(and_, f_exprs)
    else:
        return True


def get_duration(metadata: list[dict]) -> Optional[float]:
    """try to find the acquisition duraction of the photon files, and return as float only if they
    are all equal, otherwise returns `None`
    """

    durations = {m.get("acquisition_duration", None) for m in metadata}
    if None in durations:
        return None
    elif len(durations) != 1:
        return None
    return durations.pop()


def format_size(size_in_bytes: int) -> str:
    if size_in_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_in_bytes, 1000)))
    p = math.pow(1000, i)
    s = round(size_in_bytes / p, 2)
    return f"{s} {size_names[i]}"


class BurstResult(TypedDict):
    name: str
    df: pl.DataFrame
    metadata: dict


def to_treeview(nodes: list[FRETNode]) -> list[dict]:
    items = []
    for node_idx, fret_node in enumerate(nodes):
        item = {
            "name": fret_node.name.value,
            "id": str(node_idx),
            "icon": "mdi-ruler",
            "children": [
                {
                    "name": "Photons",
                    "id": f"{node_idx}:photons",
                    "icon": "mdi-lightbulb",
                    "children": [
                        {
                            "name": photon_file.name,
                            "id": f"{node_idx}:photons:{ph_idx}",
                            "icon": "mdi-file-star",
                        }
                        for ph_idx, photon_file in enumerate(fret_node.photons.items)
                    ],
                },
                {
                    "name": "Bursts",
                    "id": f"{node_idx}:bursts",
                    "icon": "mdi-flash",
                    "children": [
                        {
                            "name": burst_item.name,
                            "id": f"{node_idx}:bursts:{b_idx}",
                            "icon": "mdi-file-chart",
                        }
                        for b_idx, burst_item in enumerate(fret_node.bursts.items)
                    ],
                },
            ],
        }

        items.append(item)
    return items


def get_info(photons: PhotonData) -> dict:
    info = {}
    info["creation_time"] = photons.metadata["creation_time"]
    info["number_of_photons"] = len(photons)
    info["acquisition_duration"] = photons.metadata["acquisition_duration"]
    info["power_diode"] = photons.metadata["tags"]["UsrPowerDiode"]["value"]

    info["cps"] = photons.cps
    t_max = photons.photon_times.max()
    counts = photons.data["stream"].value_counts(sort=True)
    info["stream_cps"] = {k: v / t_max for k, v in counts.iter_rows()}

    if comment := photons.comment:
        info["comment"] = comment
    return info
