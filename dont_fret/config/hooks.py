import polars as pl

from dont_fret.expr import parse_expression
from dont_fret.models import Bursts, PhotonData


def suffice(name: str, suffix: str = "") -> str:
    if suffix:
        return f"{name}_{suffix}"
    else:
        return name


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


def with_columns(bursts: Bursts, exprs: dict[str, pl.Expr | str], suffix="") -> Bursts:
    parsed_exprs = []
    for k, v in exprs.items():
        alias = suffice(k, suffix)
        if isinstance(v, str):
            parsed_exprs.append(parse_expression(v).alias(alias))
        else:
            parsed_exprs.append(v.alias(alias))

    return bursts.with_columns(parsed_exprs)


def drop(bursts: Bursts, columns: list[str], suffix="") -> Bursts:
    return bursts.drop(columns)
