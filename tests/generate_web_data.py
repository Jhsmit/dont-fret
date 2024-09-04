"""Generates data for testing the web app"""
from pathlib import Path

from dont_fret.web.apps.burst_view import gen_fileitems
from dont_fret.web.apps.burst_view.state import PhotonFiles, BurstColors
import pandas as pd

# %%
cwd = Path(__file__).parent
OUTPUT_PATH = cwd / "test_data" / "output" / "web"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# %%
DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}

# %%

photon_file_items = gen_fileitems(2)

colors = list(BurstColors.from_dict(DCBS_TEST))
photon_files = PhotonFiles()
photon_files.add_file_items(photon_file_items)
bursts_item = photon_files.do_burst_search(
    "my_bursts", colors, on_progress=lambda x: None, on_indeterminate=lambda x: None
)

# %%


bursts_item.df.to_csv(OUTPUT_PATH / "bursts.csv", index=False)


# %%
bursts_item.df.dtypes

# %%

df_read = pd.read_csv(OUTPUT_PATH / "bursts.csv")
df_read

# %%

d = bursts_item.burst_sets["200924-FRET-MBP-2K-R0_36.ptu"].to_dict(photon_counts=True)

d["n_photons"].dtype

# %%
#
# d['n_DD']
#
# #%%
#
# b = bursts_item.burst_sets['200924-FRET-MBP-2K-R0_36.ptu']
# type(b[0].stream_counts['DD'])
#
#
# #%%
#
#
# #%%
#
# import numpy as np
# np.iinfo(np.uint8)
#
# #%%
#
# a = np.array([1,2,3, 4], dtype=np.int64)
# a
#
# #%%
# #
# b = a.astype(np.min_scalar_type(a.max()))
#
# b.base
#
# a[:2].base
