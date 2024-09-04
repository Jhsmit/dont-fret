# %%

from pathlib import Path

import numpy as np
import proplot as pplt
from KDEpy.utils import cartesian

from dont_fret.fileIO import load_bursts
from dont_fret.formatting import CONTOURF_KWARGS, HIST1D_KWARGS
from dont_fret.plot import (
    calculate_kde_2d,
    reshape_grid,
)

# %%
# __file__ = Path().resolve() / "templates" / 'script.py'  # pycharm scientific mode

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "output" / "ds1"
burst_file = test_data_dir / "dcbs_bursts.h5"
bursts = load_bursts(burst_file)

# %%

x, y = bursts.E_app, bursts.S_app
xlim = (0, 1)
ylim = (0, 1)


# %%
# KDE plot
NUM = 2**10
lin_x = np.linspace(*xlim, NUM, endpoint=True)
lin_y = np.linspace(*ylim, NUM, endpoint=True)

grid_points = cartesian([lin_x, lin_y])
grid, kde_z = calculate_kde_2d(x, y, grid_points=grid_points)
grid_x, grid_y = reshape_grid(grid, (NUM, NUM))
z = kde_z.reshape((NUM, NUM)).T

fig, ax = pplt.subplots(axwidth="60mm")
ax.contourf(grid_x, grid_y, z, **CONTOURF_KWARGS)
ax.format(xlim=xlim, ylim=ylim, xlabel="Apparent FRET", ylabel="Apparent Stoichiometry")

panel_x = ax.panel_axes("t", width="15mm")
panel_x.hist(x, **HIST1D_KWARGS)
panel_y = ax.panel_axes("r", width="15mm")
panel_y.histh(y, **HIST1D_KWARGS)
pplt.show()

# %%

# 2D histogram plot
xbins = np.linspace(*xlim, 30, endpoint=True)
ybins = np.linspace(*ylim, 30, endpoint=True)

fig, ax = pplt.subplots(axwidth="60mm")
h, xedge, yedge, img = ax.hist2d(x, y, cmap="viridis", bins=[xbins, ybins], edgefix=False)
ax.colorbar(img, loc="b", label="Counts", width="2.5mm")
ax.format(xlim=xlim, ylim=ylim, xlabel="Apparent FRET", ylabel="Apparent Stoichiometry")

kw = {k: v for k, v in HIST1D_KWARGS.items() if k != "bins"}

panel_x = ax.panel_axes("t", width="15mm")
panel_x.hist(x, bins=xbins, **kw)
panel_y = ax.panel_axes("r", width="15mm")
panel_y.histh(y, bins=ybins, **kw)
pplt.show()

# %%
# ISSUE 38
# df = bursts.to_dataframe(photon_counts=True)
# df.columns
#
# #%%
#
# # import proplot as pplt
# #
# fig, axes = pplt.subplots(nrows=2, ncols=2, sharey=False, share=False)
#
# for ax, stream in zip(axes,  bursts[0].stream_counts):
#     if stream == "AD":
#         continue
#     ax.hist(df[f"n_{stream}"], bins="fd")
#     mean = df[f"n_{stream}"].mean()
#     print(f"{stream}: {mean:.2f} photons per burst")
#     ax.axvline(mean, color="r", linestyle="--")
#     ax.format(xlabel="Photons per burst", ylabel="Count", title=stream)
#
# green_ch = df["n_DD"] + df["n_DA"]
# axes[1, 1].hist(green_ch, bins="fd")
# mean = green_ch.mean()
# print(f"DD + DA: {mean:.2f} photons per burst")
# axes[1, 1].axvline(mean, color="r", linestyle="--")
# axes[1, 1].format(xlabel="Photons per burst", ylabel="Count", title="DD + DA")
#
# pplt.show()
