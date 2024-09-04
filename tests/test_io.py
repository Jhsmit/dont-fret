from pathlib import Path

import pytest
import yaml

# from helpers import assert_burstsets_equal, assert_photons_equal
from dont_fret.fileIO import (
    PhotonFile,
    read_ptu_metadata,
)
from dont_fret.models import Bursts, PhotonData
from dont_fret.utils import clean_types

cwd = Path(__file__).parent
input_data_dir = cwd / "test_data" / "input"
output_data_dir = cwd / "test_data" / "output"


@pytest.fixture
def ph_ds1() -> PhotonData:
    return PhotonData.from_file(PhotonFile(input_data_dir / "ds1" / "datafile_1.ptu"))


@pytest.fixture
def dcbs_bursts(ph_ds1: PhotonData) -> Bursts:
    return ph_ds1.burst_search("DCBS")


def test_read_ptu_metadata():
    metadata = read_ptu_metadata(input_data_dir / "ds1" / "datafile_1.ptu")

    m_saved = yaml.safe_load((output_data_dir / "ds1" / "metadata.yaml").read_text())
    m_saved["record_type"] = "rtTimeHarp260PT3"
    m_clean = clean_types(metadata)

    assert m_saved == m_clean
