# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from dont_fret.channel_kde import compute_alex_2cde, compute_fret_2cde, convolve_stream, make_kernel
from dont_fret.expr import is_in_expr
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds3"
ptu_file = "datafile_1.ptu"
ptu_file = "210122_sFRET_MBP5K_apo_15.ptu"

# %%
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))
bursts = photons.burst_search("APBS")

tau = 50e-6
assert photons.timestamps_unit
kernel = make_kernel(tau, photons.timestamps_unit)

fig, ax = plt.subplots()
ax.plot(kernel)

# %%
# tomov et al eqn 9
D_ex = convolve_stream(photons.data, ["DD", "DA"], kernel)
A_ex = convolve_stream(photons.data, ["AA"], kernel)  # crashed the kernel (sometimes)

# %%

kde_data = photons.data.select(
    [pl.col("timestamps"), pl.col("stream"), pl.lit(D_ex).alias("D_ex"), pl.lit(A_ex).alias("A_ex")]
)
alex_2cde = compute_alex_2cde(bursts.photon_data, kde_data)

DA = convolve_stream(photons.data, ["DA"], kernel)
DD = convolve_stream(photons.data, ["DD"], kernel)
kde_data = photons.data.select(
    [pl.col("timestamps"), pl.col("stream"), pl.lit(DA).alias("DA"), pl.lit(DD).alias("DD")]
)

# %%

fret_2cde = compute_fret_2cde(bursts.indices, kde_data)

fret_2cde

# %%

fig, axes = plt.subplots(ncols=2)
axes[0].hist(alex_2cde, bins="fd")
axes[1].hist(fret_2cde, bins="fd")
# axes[1].axvline(10, color='r')
# %%
