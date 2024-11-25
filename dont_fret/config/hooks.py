from dont_fret.models import Bursts, PhotonData


def fret_2cde(
    bursts: Bursts,
    photon_data: PhotonData,
    tau: float = 50e-6,
    dem_stream: str = "DD",
    aem_stream: str = "DA",
) -> Bursts:
    return bursts.fret_2cde(photon_data, tau=tau, dem_stream=dem_stream, aem_stream=aem_stream)


def alex_2cde(
    bursts: Bursts, photon_data: PhotonData, tau: float = 50e-6, dex_streams=None, aex_streams=None
) -> Bursts:
    return bursts.alex_2cde(photon_data, tau=tau, dex_streams=dex_streams, aex_streams=aex_streams)
