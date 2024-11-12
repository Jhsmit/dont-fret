# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.testing as pl_test

from dont_fret.channel_kde import compute_alex_2cde, convolve_stream, make_kernel
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData

cwd = Path(__file__).parent
input_data_dir = cwd / "test_data" / "input"
output_data_dir = cwd / "test_data" / "output"


def test_compare_cde_fretbursts():
    tau = 50e-6
    TIMESTAMPS_UNIT = 1.25e-08

    photon_data = pl.read_parquet(output_data_dir / "kde" / "photon_data.pq")
    bursts_ref = pl.read_parquet(output_data_dir / "kde" / "burst_data.pq")

    kernel = make_kernel(tau, TIMESTAMPS_UNIT)
    D_ex = convolve_stream(photon_data, ["DD", "DA"], kernel)
    A_ex = convolve_stream(photon_data, ["AA"], kernel)
    kde_data = photon_data.select(
        [
            pl.col("timestamps"),
            pl.col("stream"),
            pl.lit(D_ex).alias("D_ex"),
            pl.lit(A_ex).alias("A_ex"),
        ]
    )

    burst_dfs = []
    for i in range(len(bursts_ref)):
        istart = bursts_ref[i]["istart"].item()
        istop = bursts_ref[i]["istop"].item()
        b_df = photon_data[istart : istop + 1].with_columns(pl.lit(i).alias("burst_index"))
        burst_dfs.append(b_df)

    burst_data = pl.concat(burst_dfs, how="vertical_relaxed")

    alex_2cde_vals = compute_alex_2cde(burst_data, kde_data)
    pl_test.assert_series_equal(alex_2cde_vals.fill_null(np.nan), bursts_ref["alex_2cde"])
