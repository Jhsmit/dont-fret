# %%

from pathlib import Path

import numpy as np
import ultraplot as uplt
from KDEpy.utils import cartesian

from dont_fret.formatting import CONTOURF_KWARGS, HIST1D_KWARGS
from dont_fret.kde_plot import (
    calculate_kde_2d,
    reshape_grid,
)
from pathlib import Path

# %%
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData
from dont_fret.process import process_photon_data
from dont_fret.config import cfg

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds1"
output_data_dir = cwd.parent / "tests" / "test_data" / "output"
ptu_file = "datafile_1.ptu"

# %%

# process photon data to get DCBS bursts
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))
bursts = process_photon_data(photons, cfg.burst_search["DCBS"])
bursts

# %%

x, y = bursts.burst_data["E_app"].to_numpy(), bursts.burst_data["S_app"].to_numpy()
xlim = (0, 1)
ylim = (0, 1)


# %%
# KDE plot
NUM = 2**8
lin_x = np.linspace(*xlim, NUM, endpoint=True)
lin_y = np.linspace(*ylim, NUM, endpoint=True)

grid_points = cartesian([lin_x, lin_y])
grid, kde_z = calculate_kde_2d(x, y, grid_points=grid_points)
grid_x, grid_y = reshape_grid(grid, (NUM, NUM))
z = kde_z.reshape((NUM, NUM)).T

fig, ax = uplt.subplots(axwidth="60mm")
ax.contourf(grid_x, grid_y, z, **CONTOURF_KWARGS)
ax.format(xlim=xlim, ylim=ylim, xlabel="Apparent FRET", ylabel="Apparent Stoichiometry", grid=False)

panel_x = ax.panel_axes("t", width="15mm", pad=0)
panel_x.hist(x, **HIST1D_KWARGS)
panel_y = ax.panel_axes("r", width="15mm")
panel_y.histh(y, **HIST1D_KWARGS)
uplt.show()

# %%

# 2D histogram plot
xbins = np.linspace(*xlim, 30, endpoint=True)
ybins = np.linspace(*ylim, 30, endpoint=True)

fig, ax = uplt.subplots(axwidth="60mm")
h, xedge, yedge, img = ax.hist2d(x, y, cmap="viridis", bins=[xbins, ybins], edgefix=False)
ax.colorbar(img, loc="b", label="Counts", width="2.5mm")
ax.format(xlim=xlim, ylim=ylim, xlabel="Apparent FRET", ylabel="Apparent Stoichiometry")

kw = {k: v for k, v in HIST1D_KWARGS.items() if k != "bins"}

panel_x = ax.panel_axes("t", width="15mm")
panel_x.hist(x, bins=xbins, **kw)
panel_y = ax.panel_axes("r", width="15mm")
panel_y.histh(y, bins=ybins, **kw)
uplt.show()
