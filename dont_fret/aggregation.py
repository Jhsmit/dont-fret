from functools import wraps
from typing import Callable, Iterable, Optional

import polars as pl

from dont_fret.expr import parse_expression
from dont_fret.models import Bursts, PhotonData
from dont_fret.utils import suffice

# photon aggregation hooks


# Global registry of transforms
aggregation_registry: dict[str, Callable] = {}


def aggregate(_func=None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator to register a aggregate function.

    Can be used as @aggregate or @aggregate(name="custom_name")
    """

    def decorator(func: Callable) -> Callable:
        agg_name = name or func.__name__
        aggregation_registry[agg_name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


@aggregate
def length(name="n_photons", suffix: str = "") -> list[pl.Expr]:
    return [pl.len().alias(suffice(name, suffix))]


@aggregate
def stream_length(streams: Iterable[str], suffix: str = "") -> list[pl.Expr]:
    """length of each stream (number of elements)"""
    return [
        (pl.col("stream") == stream).sum().alias(suffice(f"n_{stream}", suffix))
        for stream in streams
    ]


@aggregate
def stream_mean(streams: Iterable[str], column: str, suffix: str = "") -> list[pl.Expr]:
    return [
        pl.col(column)
        .filter(pl.col("stream") == stream)
        .mean()
        .alias(suffice(f"{column}_{stream}", suffix))
        for stream in streams
    ]


@aggregate
def column_stats(
    column: str, stat_funcs: list[str] = ["mean", "min", "max"], suffix: str = ""
) -> list[pl.Expr]:
    return [
        getattr(pl.col(column), d)().alias(suffice(f"{column}_{d}", suffix)) for d in stat_funcs
    ]


@aggregate
def parse_expr(exprs: dict[str, str], suffix: str = "") -> list[pl.Expr]:
    return [parse_expression(v).alias(suffice(k, suffix)) for k, v in exprs.items()]


@aggregate
def stream_asymmetry(
    lhs_streams: list[str], rhs_streams: list[str], column: str, suffix=""
) -> list[pl.Expr]:
    value_lhs = pl.col(column).filter(pl.col("stream").is_in(lhs_streams)).mean()
    value_rhs = pl.col(column).filter(pl.col("stream").is_in(rhs_streams)).mean()

    return [(value_lhs - value_rhs).alias(suffice("asymmetry", suffix))]
