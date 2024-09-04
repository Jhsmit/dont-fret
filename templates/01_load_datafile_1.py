# %%
from pathlib import Path

from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds1"
ptu_file = "datafile_1.ptu"

# %%
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))


# %%

bursts = photons.burst_search("DCBS")

# %%

# Export to a dataframe, skip the number of photons but include photon count
# and mean life time per burst per photon stream.

bursts.burst_data.head()
print(bursts.burst_data)
