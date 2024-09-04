"""
Generates reference burst search results for tests
"""

from pathlib import Path

from dont_fret.fileIO import PTUFile, save
from dont_fret.models import ChannelPhotonData

cwd = Path(__file__).parent

DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}

input_data_dir = cwd / "test_data" / "input" / "ds1"
output_data_dir = cwd / "test_data" / "output" / "ds1"
output_data_dir.mkdir(exist_ok=True, parents=True)

photons = ChannelPhotonData.from_file(PTUFile(input_data_dir / "datafile_1.ptu"))

save(output_data_dir / "photons.h5", photons)

bursts_dcbs = photons.burst_search(DCBS_TEST).filter("n_photons", (50, None))
bursts_dcbs.to_dataframe().write_csv(output_data_dir / "dcbs_bursts.csv")
save(output_data_dir / "dcbs_bursts.h5", bursts_dcbs)

bursts_apbs = photons.burst_search(APBS_TEST).filter("n_photons", (50, None))
bursts_apbs.to_dataframe().write_csv(output_data_dir / "apbs_bursts.csv")
save(output_data_dir / "apbs_bursts.h5", bursts_apbs)
