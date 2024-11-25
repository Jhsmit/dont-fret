"""
Module for channel based kernel density estimation

This is based on:
Disentangling Subpopulations in Single-Molecule FRET and ALEX Experiments with Photon Distribution Analysis
https://doi.org/10.1016/j.bpj.2011.11.4025

Please cite the paper when using this module.

"""

import warnings
from typing import Optional

import numpy as np
import polars as pl
from numba import float64, int64, jit, types

from dont_fret.expr import is_in_expr


def compute_fret_2cde(
    burst_photons: pl.DataFrame,
    kde_rates: pl.DataFrame,
    dem_stream: str = "DD",
    aem_stream: str = "DA",
) -> np.ndarray:
    """
    burst_photons: Dataframe with columns: timestamps, stream, burst_index
    kde_rates: Dataframe with columns: timestamps, D_em, A_em.
    dem_stream: photon stream which is donor emission (default: DD)
    aem_stream: photon stream which is acceptor emission (default: DA)
    """
    joined_df = burst_photons.join(kde_rates, on=["timestamps"], how="left").filter(
        pl.col("stream") == pl.col("stream_right")
    )

    f_DA = pl.col("stream") == aem_stream
    f_DD = pl.col("stream") == dem_stream
    df_DA = joined_df.filter(f_DA)
    df_DD = joined_df.filter(f_DD)

    matching_indices = np.intersect1d(df_DA["burst_index"].unique(), df_DD["burst_index"].unique())

    DA_groups = {k: v for k, v in df_DA.group_by("burst_index")}
    DD_groups = {k: v for k, v in df_DD.group_by("burst_index")}

    N: int = burst_photons["burst_index"].max() + 1  # type: ignore
    output = np.full(N, fill_value=np.nan)
    for j in matching_indices:
        df_DA_j = DA_groups[(j,)]
        df_DD_j = DD_groups[(j,)]

        kde_DA_DA = df_DA_j["A_em"].to_numpy()  # select DA density - kde^DA_DA
        kde_DA_DD = df_DD_j["A_em"].to_numpy()  # kde^DA_DD (in the paper called kde^A_D)
        kde_DD_DD = df_DD_j["D_em"].to_numpy()  # kde^DD_DD
        kde_DD_DA = df_DA_j["D_em"].to_numpy()  # kde^DD_DA

        try:
            nbkde_DA_DA = (1 + 2 / len(kde_DA_DA)) * (kde_DA_DA - 1)
            nbkde_DD_DD = (1 + 2 / len(kde_DD_DD)) * (kde_DD_DD - 1)

            # when denom is zero, it doesnt count towards number of photons
            # see "Such cases are removed by the computer algorithm", in Tomov et al.
            denom = kde_DA_DD + nbkde_DD_DD
            ED = (kde_DA_DD / denom).sum() / np.count_nonzero(denom)  # when

            denom = kde_DD_DA + nbkde_DA_DA
            EA = (kde_DD_DA / denom).sum() / np.count_nonzero(denom)  # = (1 - E)_A

            fret_cde = 110 - 100 * (ED + EA)
            output[j] = fret_cde
        except ZeroDivisionError:
            output[j] = np.nan
    return output


# refactor to indices / loop
def compute_alex_2cde(
    burst_photons: pl.DataFrame,
    kde_rates: pl.DataFrame,
    dex_streams: Optional[list[str]] = None,
    aex_streams: Optional[list[str]] = None,
) -> pl.Series:
    """
    burst_photons: Dataframe with columns: timestamps, stream, burst_index
    kde_rates: Dataframe with columns: timestamps, D_ex, A_ex.
    dex_streams: list of photon streams which are donor excitation (default: DD, DA)
    aex_streams: list of photon streams which are acceptor excitation (default: AA)
    """
    dex_streams = dex_streams if dex_streams else ["DD", "DA"]
    aex_streams = aex_streams if aex_streams else ["AA"]

    f_dex = is_in_expr("stream", dex_streams)
    f_aex = is_in_expr("stream", aex_streams)

    # equivalent to (but faster):
    # joined_df = burst_photons.join(kde_rates, on=['timestamps', 'stream'], how='inner')
    # still, this is the slowest step. would be nice if we can improve since we know the indices

    joined_df = burst_photons.join(kde_rates, on=["timestamps"], how="left").filter(
        pl.col("stream") == pl.col("stream_right")
    )

    b_df = joined_df.select(
        [
            pl.col("burst_index"),
            pl.col("stream"),
            (pl.col("A_ex") / pl.col("D_ex")).alias("ratio_AD"),
        ]
    )

    # tomov et al eqn 10 and 11
    df_f_dex = b_df.filter(f_dex)
    agg_dex = df_f_dex.group_by("burst_index", maintain_order=True).agg(
        [pl.col("ratio_AD").sum(), pl.len().alias("N_dex")]
    )

    df_f_aex = b_df.filter(f_aex)
    agg_dax = df_f_aex.group_by("burst_index", maintain_order=True).agg(
        [(1 / pl.col("ratio_AD")).sum().alias("ratio_DA"), pl.len().alias("N_aex")]
    )

    combined = pl.concat([agg_dex, agg_dax], how="align")

    # tomov et al eqn 12
    # this is an addition in the paper
    # in fretbursts its subtracted
    ax_2cde_bracket = (1 / pl.col("N_aex")) * pl.col("ratio_AD") + (1 / pl.col("N_dex")) * pl.col(
        "ratio_DA"
    )

    ax_2cde_norm = pl.lit(100) - pl.lit(50) * ax_2cde_bracket

    alex_2cde = combined.select(ax_2cde_norm.alias("alex_2cde")).to_series()

    return alex_2cde


def make_kernel(
    tau: float, timestamps_unit: float, domain_size: int = 10, kernel="laplace"
) -> np.ndarray:
    window_size = domain_size * (tau / timestamps_unit)
    window_size_even_int = 2 * round(window_size / 2)

    # check that rounding error isnt too large
    rel_dev = (window_size - window_size_even_int) / window_size
    if np.abs(rel_dev) > 0.01:
        warnings.warn(
            "Kernel window size deviation from rounding larger than 1 percent. Choose a smaller `tau` with respect to `timestamps_unit'"
        )

    t_eval = np.linspace(-domain_size / 2, domain_size / 2, window_size_even_int + 1, endpoint=True)
    kernel = np.exp(-np.abs(t_eval))

    return kernel


def convolve_stream(data: pl.DataFrame, streams: list[str], kernel: np.ndarray) -> np.ndarray:
    f_expr = is_in_expr("stream", streams)

    df = data.filter(f_expr)
    # TODO warn on copy
    # dataframes read from .pq cannot be converted zero-copy
    event_times = df["timestamps"].to_numpy(allow_copy=True)
    eval_times = data["timestamps"].to_numpy(allow_copy=True)
    return async_convolve(event_times, eval_times, kernel)


@jit(
    float64[:](
        types.Array(int64, 1, "C", readonly=True),
        types.Array(int64, 1, "C", readonly=True),
        types.Array(float64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
)
def async_convolve(event_times, eval_times, kernel):
    """convolve integer timestamps with a kernel"""

    i_lower = 0
    i_upper = 0

    window_half_size = len(kernel) // 2
    result = np.zeros_like(eval_times, dtype=np.float64)

    for i, time in enumerate(eval_times):
        while event_times[i_upper] < time + window_half_size:
            i_upper += 1
            if i_upper == len(event_times):
                i_upper -= 1
                break

        while event_times[i_lower] < time - window_half_size:
            i_lower += 1
            if i_lower == len(event_times):
                i_lower -= 1
                break

        for j in range(i_lower, i_upper):
            idx = event_times[j] - time + window_half_size
            result[i] += kernel[idx]

    return result
