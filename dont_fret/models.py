from __future__ import annotations

import json
from functools import cached_property, reduce
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import polars as pl

from dont_fret.burst_search import bs_eggeling, return_intersections
from dont_fret.config.config import BurstColor, DontFRETConfig, cfg
from dont_fret.support import get_binned
from dont_fret.utils import clean_types

if TYPE_CHECKING:
    from dont_fret.fileIO import PhotonFile


class PhotonData:
    """Base object for timestamp data


    does not have identified channels
    and access the structured array with properties

    timestamps: ndarray int timestamps (global resolution)
    detectors: ndarray int
    nanotimes, optional: ndarray int

    metadata: dict; whatever contents
    """

    def __init__(
        self, data: pl.DataFrame, metadata: Optional[dict] = None, cfg: DontFRETConfig = cfg
    ):
        self.data = data
        self.metadata = metadata or {}
        self.cfg = cfg

    def __str__(self):
        s = str(self.__class__) + " object\n"
        s += f"Number of photons: {len(self)}\n"
        return s

    def __hash__(self) -> int:
        # Hashes only the metadata
        # If you butcher the metadata you might get a collision

        # TODO: combine with config and version

        s = json.dumps(clean_types(self.metadata), sort_keys=True)
        return hash(s)

    @property
    def timestamps(self) -> pl.Series:
        """Array of integer typestamps"""
        return self.data["timestamps"]

    @property
    def detectors(self) -> pl.Series:
        return self.data["detectors"]

    @property
    def nanotimes(self) -> Optional[pl.Series]:
        try:
            return self.data["nanotimes"]
        except KeyError:
            return None

    @property
    def cps(self) -> float:
        """Average count rate (counts per second / Hz)"""
        return len(self) / float(self.photon_times.max())  # type: ignore

    def __getitem__(self, key):
        # TODO sort when slicing backwards?
        return self.__class__(self.data[key], metadata=self.metadata, cfg=self.cfg)

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_file(cls, f_obj: PhotonFile, metadata: Optional[dict] = None):
        data, f_metadata = f_obj.load_file()
        full_metadata = {**f_metadata, **(metadata or {})}
        if "timestamps_unit" not in full_metadata.keys():
            raise ValueError("Missing field 'timestamps_unit' in metadata")
        if f_obj.filename:
            full_metadata["filename"] = f_obj.filename

        return cls(data, metadata=full_metadata)

    @classmethod
    def load(cls, directory: Path) -> PhotonData:
        data = pl.read_parquet(directory / "data.pq")
        with open(directory / "metadata.json", "r") as f:
            metadata = json.load(f)

        cfg = DontFRETConfig.from_yaml(directory / "config.yaml")
        return PhotonData(data, metadata, cfg)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.data.write_parquet(directory / "data.pq")
        with open(directory / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        self.cfg.to_yaml(directory / "config.yaml")

    @property
    def monotonic(self) -> bool:
        """Is `True` is the timestamps are monotonically increasing"""
        return self.timestamps.diff().ge(0).all(ignore_nulls=True)

    @property
    def timestamps_unit(self) -> Optional[float]:
        """Multiplication factor to covert timestamps integers to seconds"""
        try:
            return self.metadata["timestamps_unit"]
        except KeyError:
            return None

    @property
    def nanotimes_unit(self) -> Optional[float]:
        """Multiplication factor to covert nanotimes integers to seconds"""
        try:
            return self.metadata["nanotimes_unit"]
        except KeyError:
            return None

    @property
    def tau_mean(self) -> float:
        """Mean of the nanotimes (in seconds). Sometimes referred to as 'fast lifetime' as it is fast to compute"""
        if self.nanotimes is not None and self.nanotimes_unit is not None:
            return self.nanotimes.mean() * self.nanotimes_unit
        else:
            return np.nan

    @property
    def photon_times(self) -> pl.Series:
        """Photon arrival times in seconds (without nanotime)"""
        return self.timestamps * self.timestamps_unit

    @property
    def n_photons(self) -> int:
        """Number of photons"""
        return len(self)

    @property
    def tmax(self) -> float:
        """Last timepoint in seconds"""
        return self.timestamps[-1] * self.timestamps_unit

    @property
    def tmin(self) -> float:
        """First timepoint in seconds"""
        return self.timestamps[0] * self.timestamps_unit

    @property
    def description(self) -> str:
        """User-defined description of the file"""
        s = ""
        s += f"Datetime: {self.metadata['tags']['File_CreatingTime']['value']}\n"
        s += f"Duration: {self.metadata['acquisition_duration']}\n"
        s += f"Power diode: {self.metadata['tags']['UsrPowerDiode']['value']:.2f}\n"

        if self.comment:
            s += "Comment:\n"
            s += self.comment

        return s

    @property
    def comment(self) -> str:
        """User-defined comment"""
        try:
            return self.metadata["tags"]["File_Comment"]["data"]
        except KeyError:
            return ""

    @property
    def measurement_type(self) -> str:
        """
        Photon-hdf5 qualifier

        Returns:
            String of the measurement type

        """
        return self.metadata["measurement_type"]

    def to_file(self):
        """write to photon-hdf5 file"""
        ...

    def burst_search(self, colors: Union[str, list[BurstColor]]) -> Bursts:
        """
        Search for bursts in the photon data.

        The burst search supports N-'color' burst search. For example, the following `colors`
        can be used for a typical dual color burst search:

        >>> colors = [
        ...     BurstColor(streams=['DD', 'DA'], L=35, M=50, T=0.0005),
        ...     BurstColor(streams=['AA'], L=30, M=50, T=0.0005)
        ... ]

        Burst seach will be applied to each stream separately, after which overlapping intervals
        between both streams are determined which yields the returned set of bursts.

        Args:
            search_spec: Dictionary specifying the burst search parameters for each stream.

        Returns:
            BurstSet object containing the identified bursts.

        """

        if isinstance(colors, str):
            burst_colors = cfg.burst_search[colors]
        elif isinstance(colors, list):
            burst_colors = colors
        else:
            raise ValueError("colors must be a string or list of BurstColor objects")

        if self.timestamps_unit is None:
            raise ValueError("timestamps_unit must be set before searching for bursts")

        # Create a list of timestamp tuples marking start and end of the bursts
        # Final output is a list of lists of tuples with one sublist of tuples per burst photon stream
        times_list = []
        for c in burst_colors:
            timestamps = self.data.filter(pl.col("stream").is_in(c.streams))[
                "timestamps"
            ].to_numpy()
            T_int = np.round(c.T / (2 * self.timestamps_unit)).astype("int32")

            # Indices of timestamps of burst time start and stop
            indices = bs_eggeling(timestamps, c.L, c.M, T_int)
            # Convert indices to times
            # TODO: return timestamps directly
            times = [(timestamps[imin], timestamps[imax]) for imin, imax in indices]
            times_list.append(times)

        # Check if any of the times _items is empty, if so, bursts is empty
        if any(len(t) == 0 for t in times_list):
            burst_photons = pl.DataFrame({k: [] for k in self.data.columns + ["burst_index"]})
        else:
            # Take the intersection of the time intervals found by the multi-color burst search
            final_times = reduce(return_intersections, times_list)

            if len(final_times) == 0:  # No overlap found
                burst_photons = pl.DataFrame({k: [] for k in self.data.columns + ["burst_index"]})
            else:
                tmin, tmax = np.array(final_times).T

                # Convert back to indices
                imin = np.searchsorted(self.timestamps, tmin)
                imax = np.searchsorted(self.timestamps, tmax)
                # take all photons (up to and including? edges need to be checked!)
                b_num = int(2 ** np.ceil(np.log2((np.log2(len(imin))))))
                dtype = getattr(pl, f"UInt{b_num}", pl.Int32)
                bursts = [
                    self.data[i1 : i2 + 1].with_columns(pl.lit(bi).alias("burst_index").cast(dtype))
                    for bi, (i1, i2) in enumerate(zip(imin, imax))
                ]
                burst_photons = pl.concat(bursts)

        bs = Bursts(burst_photons, metadata=self.metadata)

        return bs


class BinnedPhotonData:
    """
    Binned photon data.
    """

    def __init__(
        self,
        photon_data: PhotonData,
        binning_time: float = 1e-3,
        bounds: tuple[Optional[float], Optional[float]] = (None, None),
    ):
        # bounds interval is inclusive, exclusive

        self.photon_data = photon_data
        self.__binning_time = binning_time

        N = 10000  # max number of datapoints
        tmin = bounds[0] or 0
        tmax = bounds[1] or min(photon_data.tmax + np.spacing(1.0), N * binning_time)
        self.__bounds = (tmin, tmax)

    @property
    def binning_time(self) -> float:
        """Binning time in seconds"""
        return self.__binning_time

    @binning_time.setter
    def binning_time(self, value: float):
        self.__binning_time = value
        self.invalidate_cache()

    @property
    def bounds(self) -> tuple[float, float]:
        return self.__bounds

    @bounds.setter
    def bounds(self, value: tuple[float, float]):
        self.__bounds = value
        self.invalidate_cache()

    def invalidate_cache(self):
        if hasattr(self, "time"):
            del self.time
        if hasattr(self, "photons"):
            del self.photons

    @cached_property
    def photons(self) -> np.ndarray:  # integer array
        """Number of photons per time bin"""
        self.time, self.photons = self._get_binned()
        return self.photons

    @cached_property
    def time(self) -> np.ndarray:  # float array
        """Time axis (in seconds). Time points are centers of bins"""
        self.time, self.photons = self._get_binned()
        return self.time

    @property
    def photon_times(self) -> np.ndarray:
        """Photon arrival times in this trace, subject to `bounds`"""
        b_arr = np.array(self.bounds) / self.photon_data.timestamps_unit
        bounds_int = np.array([np.floor(b_arr[0]), np.ceil(b_arr[1])]).astype("uint64")
        i_min, i_max = np.searchsorted(self.photon_data.timestamps, bounds_int)

        return self.photon_data.photon_times[i_min:i_max]

    def _get_binned(self) -> tuple[np.ndarray, np.ndarray]:
        """Gets bin centers (time, seconds) and photons per bin"""

        return get_binned(self.photon_times.to_numpy(), self.binning_time, self.bounds)

    def __len__(self) -> int:
        """Length of the binned result"""
        return len(self.time)


class Bursts(object):
    """
    Class which holds a set of bursts.

    attrs:
    bursts np.array with Burst Objects
    """

    # todo add metadata support
    # bursts: numpy.typing.ArrayLike[Bursts] ?
    def __init__(
        self, photon_data: pl.DataFrame, metadata: Optional[dict] = None, cfg: DontFRETConfig = cfg
    ):
        self.photon_data = photon_data
        self.metadata: dict = metadata or {}
        self.cfg = cfg

        agg = [(pl.col("stream") == stream).sum().alias(f"n_{stream}") for stream in cfg.streams]
        agg.extend(
            [
                pl.when(pl.col("stream") == k).then(pl.col("nanotimes")).mean().alias(f"tau_{k}")
                for k in cfg.streams
            ]
        )

        agg.append(pl.col("timestamps").mean().alias("timestamps_mean"))
        agg.append(pl.col("timestamps").min().alias("timestamps_min"))
        agg.append(pl.col("timestamps").max().alias("timestamps_max"))

        # TODO as yaml in config
        columns = [
            (pl.col("n_DA") / (pl.col("n_DD") + pl.col("n_DA"))).alias("E_app"),
            (
                (pl.col("n_DA") + pl.col("n_DD"))
                / (pl.col("n_DD") + pl.col("n_DA") + pl.col("n_AA"))
            ).alias("S_app"),
            photon_data["burst_index"].unique_counts().alias("n_photons"),
        ]

        t_unit = self.metadata.get("timestamps_unit", None)
        if t_unit is not None:
            columns.extend(
                [
                    (pl.col("timestamps_mean") * t_unit).alias("time_mean"),
                    ((pl.col("timestamps_max") - pl.col("timestamps_min")) * t_unit).alias(
                        "time_length"
                    ),
                ]
            )

        self.burst_data = (
            self.photon_data.group_by("burst_index", maintain_order=True)
            .agg(agg)
            .with_columns(columns)
        )

    @classmethod
    def load(cls, directory: Path) -> Bursts:
        data = pl.read_parquet(directory / "data.pq")
        with open(directory / "metadata.json", "r") as f:
            metadata = json.load(f)

        cfg = DontFRETConfig.from_yaml(directory / "config.yaml")
        return Bursts(data, metadata, cfg)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.photon_data.write_parquet(directory / "data.pq")
        with open(directory / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        self.cfg.to_yaml(directory / "config.yaml")

    def __len__(self) -> int:
        """Number of bursts"""
        return len(self.burst_data)

    def __iter__(self):
        return self.photon_data.group_by("burst_index")

    @property
    def timestamps_unit(self) -> Optional[float]:
        """Multiplication factor to covert timestamps integers to seconds"""
        try:
            return self.metadata["timestamps_unit"]
        except KeyError:
            return None
