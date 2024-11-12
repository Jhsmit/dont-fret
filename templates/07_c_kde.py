# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from dont_fret.channel_kde import alex_2cde, convolve_stream, make_kernel
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds1"
ptu_file = "datafile_1.ptu"

# %%
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))

tau = 50e-6
assert photons.timestamps_unit
kernel = make_kernel(tau, photons.timestamps_unit)

fig, ax = plt.subplots()
ax.plot(kernel)

# %%

# tomov et al eqn 9
D_ex = convolve_stream(photons.data, ["DD", "DA"], kernel)
A_ex = convolve_stream(photons.data, ["AA"], kernel)

# %%

kde_data = photons.data.select(
    [pl.col("timestamps"), pl.col("stream"), pl.lit(D_ex).alias("D_ex"), pl.lit(A_ex).alias("A_ex")]
)

bursts = photons.burst_search("APBS")

# %%
alex_2cde_vals = alex_2cde(bursts.photon_data, kde_data)

# %%
