from __future__ import annotations

import itertools
import math
from functools import reduce
from operator import and_
from typing import Any, Literal, Optional, TypedDict, Union

import altair as alt
import numpy as np
import polars as pl
import solara
import solara.lab

from dont_fret.config.config import BurstColor
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.process import process_photon_data
from dont_fret.web.models import BurstFilterItem, BurstNode, FRETNode, PhotonNode


def use_dark_altair():
    dark_effective = solara.lab.use_dark_effective()
    dark_effective_previous = solara.use_previous(dark_effective)
    if dark_effective != dark_effective_previous:
        if dark_effective:
            alt.themes.enable("dark")

        else:
            alt.themes.enable("default")


def make_burst_dataframe(
    bursts: list[Bursts], names: Optional[list[str]], name_column="filename"
) -> pl.DataFrame:
    """Convert a list of `Bursts` objects into a `polars.DataFrame`."""

    concat = pl.concat([b.burst_data for b in bursts], how="vertical_relaxed")
    if names:
        lens = [len(burst) for burst in bursts]
        dtype = pl.Enum(categories=names)
        series = pl.Series(name=name_column, values=np.repeat(names, lens), dtype=dtype)

        return concat.with_columns(series)
    else:
        return concat


def make_burst_nodes(
    photon_nodes: list[PhotonNode],
    burst_settings: dict[str, list[BurstColor]],
    aggregations: Optional[dict] = None,
    transforms: Optional[dict] = None,
) -> list[BurstNode]:
    photons = [PhotonData.from_file(PhotonFile(node.file_path)) for node in photon_nodes]
    burst_nodes = []
    # todo tqdm?

    for name, burst_colors in burst_settings.items():
        bursts = [
            process_photon_data(
                photon_data, burst_colors, aggregations=aggregations, transforms=transforms
            )
            for photon_data in photons
        ]

        infos = [get_info(photons) for photons in photons]
        duration = get_duration(infos)
        df = make_burst_dataframe(bursts, names=[node.name for node in photon_nodes])
        node = BurstNode(
            name=name, df=df, colors=burst_colors, photon_nodes=photon_nodes, duration=duration
        )
        burst_nodes.append(node)

    return burst_nodes


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
