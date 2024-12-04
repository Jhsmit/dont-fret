from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from inspect import signature
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from tqdm.auto import tqdm

from dont_fret.aggregation import aggregation_registry
from dont_fret.config import CONFIG_DEFAULT, CONFIG_DEFAULT_DIR, BurstColor, DontFRETConfig, cfg
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.transform import transform_registry

# TODO pass cfg; need also to pass to `from_file`
# not sure if we should go down this road


def search_and_save(
    file: Path,
    write_photons: bool = True,
    output_type: Literal[".pq", ".csv"] = ".pq",
) -> None:
    """
    Performs burst search on the supplied file and saves burst search output to disk.
    Uses global cfg object
    """

    photons = PhotonData.from_file(PhotonFile(file))
    output_dir = file.parent
    for bs_name, burst_colors in cfg.burst_search.items():
        bursts = process_photon_data(photons, burst_colors)

        if write_photons:
            write_dataframe(
                bursts.photon_data, output_dir / f"{file.stem}_{bs_name}_photon_data{output_type}"
            )
        write_dataframe(
            bursts.burst_data, output_dir / f"{file.stem}_{bs_name}_burst_data{output_type}"
        )


def batch_search_and_save(
    files: list[Path],
    write_photons: bool = True,
    output_type: Literal[".pq", ".csv"] = ".pq",
    max_workers: Optional[int] = None,
) -> None:
    """
    Search all photon file items in batch threaded.
    """

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for f in files:
            fut = executor.submit(search_and_save, f, write_photons, output_type)
            futures.append(fut)

        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()


def write_dataframe(df: pl.DataFrame, path: Path) -> None:
    """Write a dataframe to disk. Writer used depends on path suffix."""
    if path.suffix == ".pq":
        df.write_parquet(path)
    elif path.suffix == ".csv":
        df.write_csv(path)
    else:
        raise ValueError(f"Unsupported output type: {path.suffix}")


def apply_aggregations(
    burst_photons: pl.DataFrame,
    aggregations: dict = cfg.aggregations,
) -> pl.DataFrame:
    agg_fixtures = {"streams": cfg.streams}

    agg = []
    for agg_name, agg_params in aggregations.items():
        split = agg_name.split(":", 1)
        base_name = split[0]
        suffix = split[1] if len(split) > 1 else ""

        agg_fn = aggregation_registry[base_name]
        kwargs = agg_params

        sig = signature(agg_fn)
        matching_fixtures = {k: agg_fixtures[k] for k in sig.parameters if k in agg_fixtures}
        kwargs = {**agg_params, **matching_fixtures}

        agg_expr = agg_fn(**kwargs)
        if isinstance(agg_expr, list):
            agg.extend(agg_expr)
        else:
            agg.append(agg_expr)

    burst_data = burst_photons.group_by("burst_index", maintain_order=True).agg(agg)
    return burst_data


def apply_transformations(
    bursts: Bursts,
    photon_data: PhotonData,
    transforms: dict = cfg.transforms,
) -> Bursts:
    # one global dict for fixtures?
    trs_fixtures = {"photon_data": photon_data}
    for trs_name, trs_params in transforms.items():
        split = trs_name.split(":", 1)
        base_name = split[0]

        suffix = split[1] if len(split) > 1 else ""
        transform_fn = transform_registry[base_name]
        sig = signature(transform_fn)
        matching_fixtures = {k: trs_fixtures[k] for k in sig.parameters if k in trs_fixtures}
        kwargs = {**trs_params, **matching_fixtures}

        bursts = transform_fn(bursts, suffix=suffix, **kwargs)

    return bursts


def process_photon_data(
    photon_data: PhotonData,
    burst_colors: list[BurstColor],
    aggregations: Optional[dict] = None,
    transforms: Optional[dict] = None,
) -> Bursts:
    """search and apply agg/trfms to burst data
    by default uses config from photon_data object

    """
    local_cfg_data = asdict(photon_data.cfg)
    if aggregations is not None:
        local_cfg_data["aggregations"] = aggregations
    if transforms is not None:
        local_cfg_data["transforms"] = transforms
    local_cfg = DontFRETConfig.from_dict(local_cfg_data)
    burst_photons = photon_data.burst_search(burst_colors)

    # evaluate aggregations
    burst_data = apply_aggregations(burst_photons, local_cfg.aggregations)
    bursts = Bursts(burst_data, burst_photons, photon_data.metadata, local_cfg)

    bursts = apply_transformations(bursts, photon_data, local_cfg.transforms)

    return bursts
