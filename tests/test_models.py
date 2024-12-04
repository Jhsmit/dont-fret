from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pl_test
import pytest
import yaml

from dont_fret.config import cfg
from dont_fret.config.config import BurstColor
from dont_fret.fileIO import PhotonFile
from dont_fret.models import BinnedPhotonData, Bursts, PhotonData
from dont_fret.process import process_photon_data

cwd = Path(__file__).parent
input_data_dir = cwd / "test_data" / "input"
output_data_dir = cwd / "test_data" / "output"

DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = [
    BurstColor(
        streams=["DD", "DA", "AA"],
        L=50,
        M=35,
        T=0.0005,
    )
]


@pytest.fixture
def ph_ds1() -> PhotonData:
    return PhotonData.from_file(PhotonFile(input_data_dir / "ds1" / "datafile_1.ptu"))


@pytest.fixture
def dcbs_bursts(ph_ds1: PhotonData) -> Bursts:
    bursts = process_photon_data(ph_ds1, cfg.burst_search["DCBS"])
    return bursts


@pytest.fixture
def apbs_bursts(ph_ds1: PhotonData) -> Bursts:
    bursts = process_photon_data(ph_ds1, APBS_TEST)
    return bursts


@pytest.fixture
def rnd_data() -> pl.DataFrame:
    np.random.seed(42)
    timestamps = np.cumsum(np.random.randint(20, 200, 1000))
    detectors = np.random.randint(0, 2, 1000)
    nanotimes = np.random.randint(0, 2**12 + 1, 1000)

    data = pl.DataFrame({"timestamps": timestamps, "detectors": detectors, "nanotimes": nanotimes})

    return data


def test_photon(rnd_data):
    photon_data = PhotonData(rnd_data)
    assert len(photon_data) == 1000

    sl = photon_data[:500]
    assert len(sl) == 500


def test_load_save_photons(ph_ds1: PhotonData, tmp_path: Path):
    test_dir_1 = tmp_path / "test_dir_1"
    ph_ds1.save(test_dir_1)

    photons_load = PhotonData.load(test_dir_1)
    assert ph_ds1.data.equals(photons_load.data)
    assert ph_ds1.cfg == photons_load.cfg
    assert asdict(ph_ds1.cfg) == asdict(photons_load.cfg)
    assert ph_ds1.metadata == photons_load.metadata


def test_load_save_bursts(dcbs_bursts: Bursts, tmp_path: Path):
    test_dir_1 = tmp_path / "test_dir_1"
    dcbs_bursts.save(test_dir_1)

    bursts_load = Bursts.load(test_dir_1)
    # for some reason, we require maintain_order=True on the aggregation operation
    # although the data and cfg appear to be the same
    assert dcbs_bursts.burst_data.equals(bursts_load.burst_data)
    assert dcbs_bursts.photon_data.equals(bursts_load.photon_data)
    assert dcbs_bursts.cfg == bursts_load.cfg
    assert dcbs_bursts.cfg
    assert bursts_load.cfg
    assert asdict(dcbs_bursts.cfg) == asdict(bursts_load.cfg)
    assert dcbs_bursts.metadata == bursts_load.metadata


def test_burst_search(ph_ds1: PhotonData):
    search_args = [cfg.burst_search["DCBS"], APBS_TEST]
    reference_files = ["dcbs_bursts.csv", "apbs_bursts.csv"]

    # only calculate E, S, time; no cde's
    transforms = {
        "with_columns": {
            "exprs": {
                "E_app": "n_DA / (n_DD + n_DA)",
                "S_app": "(n_DD + n_DA) / (n_DD + n_DA + n_AA)",
                "timestamps_length": "timestamps_max - timestamps_min",
            }
        }
    }

    for bs_arg, ref_file in zip(search_args, reference_files):
        bursts = process_photon_data(ph_ds1, bs_arg, transforms=transforms)
        pth = output_data_dir / "ds1" / ref_file

        df_ref = pl.read_csv(pth)
        df_test = bursts.burst_data.filter(pl.col("n_photons") > 50)

        for k in ["n_photons", "E_app", "S_app"]:
            pl_test.assert_series_equal(
                df_test[k], df_ref[k], check_dtypes=False, check_names=False
            )

        time_length = df_test["timestamps_length"] * ph_ds1.timestamps_unit
        time_min = df_test["timestamps_min"] * ph_ds1.timestamps_unit
        time_max = df_test["timestamps_max"] * ph_ds1.timestamps_unit
        pl_test.assert_series_equal(time_length, df_ref["time_length"], check_names=False)
        pl_test.assert_series_equal(time_min, df_ref["time_min"], check_names=False)
        pl_test.assert_series_equal(time_max, df_ref["time_max"], check_names=False)


def test_process_photon_data(ph_ds1: PhotonData):
    s = """
  "alex_2cde:75":
    tau: 75.e-6
  "alex_2cde:150":
    tau: 150.e-6
  fret_2cde:
    tau: 45.e-6
  with_columns:
    exprs:
      E_app: "n_DA / (n_DD + n_DA)"
      S_app: "(n_DD + n_DA) / (n_DD + n_DA + n_AA)"
      timestamps_length: "timestamps_max - timestamps_min"
    """

    transforms = yaml.safe_load(s)

    bursts = process_photon_data(ph_ds1, APBS_TEST, transforms=transforms)
    assert "alex_2cde_75" in bursts.burst_data.columns
    assert "alex_2cde_150" in bursts.burst_data.columns


def test_binning(ph_ds1):
    trace = BinnedPhotonData(ph_ds1, bounds=(0, ph_ds1.tmax))
    assert len(trace) == 81439
    trace.binning_time = 1e-2

    # check if all photons are accounted for, last photon is omitted because of
    # exclusive upper bound
    assert trace.photons.sum() == len(trace.photon_data.timestamps) - 1
    assert len(trace) == 8144
    trace.binning_time = 1e-3
    assert len(trace) == 81439

    trace = BinnedPhotonData(ph_ds1)
    assert len(trace) == 10000
    trace.binning_time = 1e-2
    assert len(trace) == 1000
    trace.bounds = (10.0, 20.0)

    assert len(trace) == 1000
    assert trace.time.min() == 10 + 1e-2 / 2


BURST_ATTRS = [
    "n_photons",
    "E_app",
    "S_app",
    "time_length",
    "time_mean",
    "time_min",
    "time_max",
]
