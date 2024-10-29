from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import and_
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, TypedDict, Union

import polars as pl

from dont_fret import BinnedPhotonData
from dont_fret.config.config import BurstColor
from dont_fret.formatting import TRACE_COLORS
from dont_fret.web.models import (
    BurstFilterItem,
    BurstNode,
    PhotonData,
    PhotonNode,
    TraceSettings,
)

if TYPE_CHECKING:
    from dont_fret.web.new_models import FRETNode


def batch_burst_search(
    photon_file_items: list[PhotonNode], burst_colors: str, max_workers: int = 4
) -> BurstNode:
    """
    Search all photon file items in batch threaded.
    """
    raise DeprecationWarning("use data manager")
    dtype = pl.Enum([item.name for item in photon_file_items])

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for f in photon_file_items:
            fut = executor.submit(burst_search, f, burst_colors, dtype)
            futures.append(fut)

    df = pl.concat([f.result() for f in futures], how="vertical_relaxed")
    metadata = {fi.name: fi.get_info() for fi in photon_file_items}

    return BurstNode(name=burst_colors, df=df, metadata=metadata)


def burst_search(
    ph_file_item: PhotonNode, burst_colors: str | list[BurstColor], dtype: pl.Enum
) -> pl.DataFrame:
    raise DeprecationWarning("use data manager instead")
    photons = ph_file_item.get_photons()
    bursts = photons.burst_search(burst_colors)

    t_unit = photons.timestamps_unit
    with_columns = [
        # (pl.col("timestamps_mean") * t_unit).alias("time_mean"),
        # ((pl.col("timestamps_max") - pl.col("timestamps_min")) * t_unit).alias("time_length"),
        pl.lit(ph_file_item.name).alias("filename").cast(dtype),
    ]

    drop_columns = ["timestamps_mean", "timestamps_min", "timestamps_max"]
    df = bursts.burst_data.with_columns(with_columns).drop(drop_columns)

    return df


def chain_filters(filters: list[BurstFilterItem]) -> Union[pl.Expr, Literal[True]]:
    """Chain a list of `BurstFilterItem` objects into a single `pl.Expr` object."""
    f_exprs = list(itertools.chain(*(f.as_expr() for f in filters if f.active)))
    if f_exprs:
        return reduce(and_, f_exprs)
    else:
        return True


# todo move to `dev`
def create_file_items(pth: Path) -> list[PhotonNode]:
    """Return a list of `PhotonFileItem` objects from a directory containing ptu files."""
    return [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]


# move to `dev` ?
def gen_fileitems(n: Optional[int] = None, directory: str = "ds2") -> list[PhotonNode]:
    """Returns a list of the first `n` FileItem objects generated from
    the data in `TEST_FILE_DIR`"""

    # root = Path(__file__).parent.parent.parent.parent
    root = Path(*Path(__file__).parts[:5])

    test_dir = root / "tests" / "test_data" / "input" / directory

    file_items = create_file_items(test_dir)
    if n is None:
        return file_items
    else:
        return file_items[:n]


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


def generate_traces(
    photons: PhotonData, trace_settings: TraceSettings
) -> dict[str, BinnedPhotonData]:
    t_bin = trace_settings.t_bin * 1e-3
    bounds = (trace_settings.t_min, trace_settings.t_max)

    traces = {}
    for stream in TRACE_COLORS:
        stream_data = PhotonData(
            photons.data.filter(pl.col("stream") == stream),
            metadata=photons.metadata,
            cfg=photons.cfg,
        )

        traces[stream] = BinnedPhotonData(stream_data, binning_time=t_bin, bounds=bounds)

    return traces


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
