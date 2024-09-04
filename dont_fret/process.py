from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from tqdm.auto import tqdm

from dont_fret.config import cfg
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData


def search_and_save(
    file: Path,
    burst_colors: str | list[str] | None = None,
    write_photons: bool = True,
    output_type: Literal[".pq", ".csv"] = ".pq",
) -> None:
    """
    Performs burst search on the supplied file and saves burst search output to disk.
    """
    photons = PhotonData.from_file(PhotonFile(file))
    if burst_colors is None:
        colors = cfg.burst_search.keys()
    elif isinstance(burst_colors, str):
        colors = [burst_colors]
    elif isinstance(burst_colors, list):
        colors = burst_colors
    output_dir = file.parent
    for color in colors:
        bursts = photons.burst_search(color)
        if write_photons:
            write_dataframe(
                bursts.photon_data, output_dir / f"{file.stem}_{color}_photon_data{output_type}"
            )
        write_dataframe(
            bursts.burst_data, output_dir / f"{file.stem}_{color}_burst_data{output_type}"
        )


def write_dataframe(df: pl.DataFrame, path: Path) -> None:
    """Write a dataframe to disk. Writier used depends on path suffix."""
    if path.suffix == ".pq":
        df.write_parquet(path)
    elif path.suffix == ".csv":
        df.write_csv(path)
    else:
        raise ValueError(f"Unsupported output type: {path.suffix}")


def batch_search_and_save(
    files: list[Path],
    burst_colors: str | list[str] | None = None,
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
            fut = executor.submit(search_and_save, f, burst_colors, write_photons, output_type)
            futures.append(fut)

        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()
