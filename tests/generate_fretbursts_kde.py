"""
generates test data for comparing fretbursts / dontfret channel kde

ran with fretbursts '0.8.3'

"""
# %%

import polars as pl
from fretbursts import *
from fretbursts.phtools import phrates

# %%


url = "http://files.figshare.com/2182601/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5"
download_file(url, save_dir="./data")

filename = "data/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5"

d = loader.photon_hdf5(filename)
loader.alex_apply_period(d)
d.calc_bg(fun=bg.exp_fit, time_s=20, tail_min_us="auto", F_bg=1.7)
d.burst_search()

ds1 = d.select_bursts(select_bursts.size, th1=30)
ds = ds1.select_bursts(select_bursts.naa, th1=30)

# %%
ph = d.ph_times_m[0]


tau_s = 50e-6  # in seconds
tau = int(tau_s / d.clk_p)  # in raw timestamp units
tau

# %%

# %%
bursts = ds1.mburst[0]

ph_dex = d.get_ph_times(ph_sel=Ph_sel(Dex="DAem"))
ph_aex = d.get_ph_times(ph_sel=Ph_sel(Aex="Aem"))

mask_dex = d.get_ph_mask(ph_sel=Ph_sel(Dex="DAem"))
mask_aex = d.get_ph_mask(ph_sel=Ph_sel(Aex="Aem"))

KDE_DexTi = phrates.kde_laplace(ph_dex, tau, time_axis=ph)
KDE_AexTi = phrates.kde_laplace(ph_aex, tau, time_axis=ph)

# %%


# %%

ALEX_2CDE = []
BRDex, BRAex = [], []
for ib, burst in enumerate(bursts):
    burst_slice = slice(int(burst.istart), int(burst.istop) + 1)
    if ~mask_dex[burst_slice].any() or ~mask_aex[burst_slice].any():
        # Either D or A photon stream has no photons in current burst,
        # thus ALEX_2CDE cannot be computed.
        ALEX_2CDE.append(np.nan)
        continue

    kde_dexdex = KDE_DexTi[burst_slice][mask_dex[burst_slice]]
    kde_aexdex = KDE_AexTi[burst_slice][mask_dex[burst_slice]]
    N_chaex = mask_aex[burst_slice].sum()
    BRDex.append(np.sum(kde_aexdex / kde_dexdex) / N_chaex)

    kde_aexaex = KDE_AexTi[burst_slice][mask_aex[burst_slice]]
    kde_dexaex = KDE_DexTi[burst_slice][mask_aex[burst_slice]]
    N_chdex = mask_dex[burst_slice].sum()
    BRAex.append(np.sum(kde_dexaex / kde_aexaex) / N_chdex)

    alex_2cde = 100 - 50 * (BRDex[-1] - BRAex[-1])
    ALEX_2CDE.append(alex_2cde)
ALEX_2CDE = np.array(ALEX_2CDE)

ALEX_2CDE

# %%

timestamps = ph
timestamps
# %%

stream = np.empty_like(timestamps, dtype="U2")
stream


streams = {
    "DD": {"Dex": "Dem"},
    "DA": {"Dex": "Aem"},
    "AA": {"Aex": "Aem"},
    "AD": {"Aex": "Dem"},
}

for stream_label, kwargs in streams.items():
    mask = d.get_ph_mask(ph_sel=Ph_sel(**kwargs))
    stream[mask] = stream_label

# %%


photons_export = pl.DataFrame({"timestamps": timestamps, "stream": stream})
photons_export.write_parquet("photon_data.pq")

# %%


burst_export = pl.DataFrame([{"istart": b.istart, "istop": b.istop} for b in bursts])
burst_export = burst_export.with_columns(pl.lit(ALEX_2CDE).alias("alex_2cde"))
burst_export.write_parquet("burst_data.pq")
