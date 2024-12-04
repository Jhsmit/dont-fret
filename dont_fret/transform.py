# transforms.py
from functools import wraps
from typing import Callable, Optional

import polars as pl

from dont_fret.expr import parse_expression
from dont_fret.models import Bursts, PhotonData
from dont_fret.utils import suffice

# Global registry of transforms
transform_registry: dict[str, Callable] = {}


TIME_UNITS = {
    "ps": 1e12,
    "ns": 1e9,
    "us": 1e6,
    "ms": 1e3,
    "s": 1,
    "min": 60,
    "h": 3600,
}


def transform(_func=None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator to register a transform function.

    Can be used as @transform or @transform(name="custom_name")
    """

    def decorator(func: Callable) -> Callable:
        transform_name = name or func.__name__
        transform_registry[transform_name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


@transform
def fret_2cde(
    bursts: Bursts,
    photon_data: PhotonData,
    tau: float = 45e-6,
    dem_stream: str = "DD",
    aem_stream: str = "DA",
    suffix="",
) -> Bursts:
    alias = suffice("fret_2cde", suffix)
    return bursts.fret_2cde(
        photon_data, tau=tau, dem_stream=dem_stream, aem_stream=aem_stream, alias=alias
    )


@transform
def alex_2cde(
    bursts: Bursts,
    photon_data: PhotonData,
    tau: float = 75e-6,
    dex_streams=None,
    aex_streams=None,
    suffix="",
) -> Bursts:
    alias = suffice("alex_2cde", suffix)

    return bursts.alex_2cde(
        photon_data,
        tau=tau,
        dex_streams=dex_streams,
        aex_streams=aex_streams,
        alias=alias,
    )


@transform
def convert_nanotimes(
    bursts: Bursts, time_unit: str = "ns", keep_columns: bool = False, suffix: str = ""
) -> Bursts:
    assert bursts.metadata
    timestamps_unit = bursts.metadata["nanotimes_unit"]
    factor = timestamps_unit * TIME_UNITS[time_unit]

    burst_data = bursts.burst_data
    timestamps_columns = [col for col in burst_data.columns if col.startswith("nanotimes")]
    conversion = [
        (pl.col(col) * factor).alias(suffice(col.replace("nanotimes", "tau"), suffix))
        for col in timestamps_columns
    ]
    burst_data = burst_data.with_columns(conversion)
    if not keep_columns:
        burst_data = burst_data.drop(timestamps_columns)

    return Bursts(burst_data, bursts.photon_data, bursts.metadata, bursts.cfg)


@transform
def convert_timestamps(
    bursts: Bursts, time_unit: str = "s", keep_columns: bool = False, suffix: str = ""
) -> Bursts:
    assert bursts.metadata
    timestamps_unit = bursts.metadata["timestamps_unit"]
    factor = timestamps_unit * TIME_UNITS[time_unit]

    burst_data = bursts.burst_data
    timestamps_columns = [col for col in burst_data.columns if col.startswith("timestamps")]
    conversion = [
        (pl.col(col) * factor).alias(suffice(col.replace("timestamps", "time"), suffix))
        for col in timestamps_columns
    ]
    burst_data = burst_data.with_columns(conversion)
    if not keep_columns:
        burst_data = burst_data.drop(timestamps_columns)

    return Bursts(burst_data, bursts.photon_data, bursts.metadata, bursts.cfg)


@transform
def with_columns(bursts: Bursts, exprs: dict[str, pl.Expr | str], suffix="") -> Bursts:
    parsed_exprs = []
    for k, v in exprs.items():
        alias = suffice(k, suffix)
        if isinstance(v, str):
            parsed_exprs.append(parse_expression(v).alias(alias))
        else:
            parsed_exprs.append(v.alias(alias))

    return bursts.with_columns(parsed_exprs)


@transform
def drop(bursts: Bursts, columns: list[str], suffix="") -> Bursts:
    return bursts.drop(columns)
