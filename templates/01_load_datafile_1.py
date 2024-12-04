# %%

# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

# %%
from dont_fret.config import CONFIG_DEFAULT_DIR, cfg, update_config_from_yaml
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData
from dont_fret.process import process_photon_data

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds1"
output_data_dir = cwd.parent / "tests" / "test_data" / "output"
ptu_file = "datafile_1.ptu"

# %%
# %%
# select a config
update_config_from_yaml(CONFIG_DEFAULT_DIR / "default_web.yaml")

# process photon data to get DCBS bursts
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))
bursts = process_photon_data(photons, cfg.burst_search["DCBS"], cfg.aggregations, cfg.transforms)
