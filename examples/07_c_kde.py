# %%

"""
Example script of how to perform ALEX and FRET photon distribution analysis (2CDE) on photon data.

Please cite the following paper when using this module:
Disentangling Subpopulations in Single-Molecule FRET and ALEX Experiments with Photon Distribution Analysis
https://doi.org/10.1016/j.bpj.2011.11.4025

"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from dont_fret.channel_kde import compute_alex_2cde, compute_fret_2cde, convolve_stream, make_kernel
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.process import apply_aggregations
from dont_fret.config import cfg
from dont_fret.expr import E_APP, S_APP
import ultraplot as uplt

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds2"
ptu_file = "f1.ptu"

# %%
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))
burst_photons = photons.burst_search("APBS")

# aggregate burst photons into bursts
# this uses cfg.aggregations by default
burst_data = apply_aggregations(burst_photons)
bursts = Bursts(burst_data, burst_photons, photons.metadata, cfg).with_columns([E_APP, S_APP])
bursts.burst_data.columns
# %%
# create a laplace kernel for KDE
tau = 50e-6
assert photons.timestamps_unit
kernel = make_kernel(tau, photons.timestamps_unit)

fig, ax = plt.subplots()
ax.plot(kernel)

# %%

# Perform kernel density estimation by convolving selected photon streams with the kernel
D_ex = convolve_stream(photons.data, ["DD", "DA"], kernel)
A_ex = convolve_stream(photons.data, ["AA"], kernel)

# %%

# Create a dataframe with convolved data, rename DA and DD streams to D_ex and A_ex
# which is expected by compute_alex_2cde
kde_data = photons.data.select(
    [pl.col("timestamps"), pl.col("stream"), pl.lit(D_ex).alias("D_ex"), pl.lit(A_ex).alias("A_ex")]
)

# %%
alex_2cde = compute_alex_2cde(burst_photons, kde_data)

# %%
DA = convolve_stream(photons.data, ["DA"], kernel)
DD = convolve_stream(photons.data, ["DD"], kernel)

# Create a dataframe with convolved data, rename DA and DD streams to A_em and D_em
# which is expected by compute_fret_2cde
kde_data = photons.data.select(
    [pl.col("timestamps"), pl.col("stream"), pl.lit(DA).alias("A_em"), pl.lit(DD).alias("D_em")]
)

fret_2cde = compute_fret_2cde(burst_photons, kde_data)
fret_2cde.shape

# add computed columns to bursts
bursts = bursts.with_columns(
    [
        pl.Series("alex_2cde", alex_2cde),
        pl.Series("fret_2cde", fret_2cde),
    ]
)

# %%

fig, axes = plt.subplots(ncols=2)
h = axes[0].hist(alex_2cde, bins="fd")
h = axes[1].hist(fret_2cde, bins="fd")

# %%
# create a 2D histogram of E_app vs FRET-2CDE
df = bursts.burst_data.select(["E_app", "fret_2cde"]).drop_nans()
x = df["E_app"].to_numpy()
y = df["fret_2cde"].to_numpy()

fig, ax = uplt.subplots(aspect=1)
h = ax.hist2d(x, y, bins=50)
ax.format(xlim=(0, 1), xlabel="Apparent FRET", ylabel="FRET-2CDE")
# %%
# create a 2D histogram of E_app vs ALEX-2CDE
df = bursts.burst_data.select(["E_app", "alex_2cde"]).drop_nans()
x = df["E_app"].to_numpy()
y = df["alex_2cde"].to_numpy()

fig, ax = uplt.subplots(aspect=1)
h = ax.hist2d(x, y, bins=50)
ax.format(xlim=(0, 1), xlabel="Apparent FRET", ylabel="ALEX-2CDE")
# %%
