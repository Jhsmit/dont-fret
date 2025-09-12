# %%

from pathlib import Path

# import proplot as pplt
import matplotlib.pyplot as pplt
import numpy as np
import polars as pl

from dont_fret.fileIO import PhotonFile
from dont_fret.formatting import TRACE_COLORS, TRACE_SIGNS
from dont_fret.models import BinnedPhotonData, PhotonData

# %%
cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds1"
ptu_file = "datafile_1.ptu"

# %%
photons = PhotonData.from_file(PhotonFile(test_data_dir / ptu_file))

fig, ax = pplt.subplots(
    # refaspect=3.5, axwidth="140mm"
)
xlim = (10, 20)
for stream in TRACE_COLORS:
    stream_photons = PhotonData(
        photons.data.filter(pl.col("stream") == stream), metadata=photons.metadata, cfg=photons.cfg
    )
    tr = BinnedPhotonData(stream_photons, bounds=xlim, binning_time=1e-3)
    ax.plot(tr.time, TRACE_SIGNS[stream] * tr.photons, color=TRACE_COLORS[stream])

ylim = np.max(np.abs(ax.get_ylim()))
# ax.format(xlabel="Time (s)", ylabel="Photons / ms", ylim=(-ylim, ylim), xlim=xlim)
pplt.show()

# %%

fig, ax = pplt.subplots(
    # refaspect=3.5, axwidth="140mm"
)
for stream in TRACE_COLORS:
    df = photons.data.filter(pl.col("stream") == stream)
    counts = np.bincount(df["nanotimes"])
    nanotime = np.arange(len(counts)) * photons.nanotimes_unit * 1e9
    ax.plot(nanotime, counts, lw=0.5, label=stream, color=TRACE_COLORS[stream])
#
# ax.format(xlabel="TCSPC time (ns)", ylabel="Photons / TCSPC bin", yscale="log")
ax.set_yscale("log")
pplt.show()

# %%
