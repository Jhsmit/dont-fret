"""
generates test data for comparing fretbursts / dontfret channel kde

ran with fretbursts '0.8.3'

adapted from https://github.com/OpenSMFS/FRETBursts/blob/master/notebooks/Example%20-%202CDE%20Method.ipynb

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

ph = d.get_ph_times(ph_sel=Ph_sel("all"))
mask_d = d.get_ph_mask(ph_sel=Ph_sel(Dex="Dem"))
mask_a = d.get_ph_mask(ph_sel=Ph_sel(Dex="Aem"))

# bursts = ds.mburst[0]
bursts = ds1.mburst[0]
# %%


def calc_fret_2cde(tau, ph, mask_d, mask_a, bursts):
    """
    Compute FRET-2CDE for each burst.

    FRET-2CDE is a quantity that tends to be around 10 for bursts which have no
    dynamics, while it has larger values (e.g. 30..100) for bursts with
    millisecond dynamics.

    References:
        Tomov et al. BJ (2012) doi:10.1016/j.bpj.2011.11.4025

    Arguments:
        tau (scalar): time-constant of the exponential KDE
        ph (1D array): array of all-photons timestamps.
        mask_d (bool array): mask for DexDem photons
        mask_a (bool array): mask for DexAem photons
        bursts (Bursts object): object containing burst data
            (start-stop indexes are relative to `ph`).

    Returns:
        FRET_2CDE (1D array): array of FRET_2CDE quantities, one element
            per burst. This array contains NaN in correspondence of bursts
            containing to few photons to compute FRET-2CDE.
    """
    # Computing KDE burst-by-burst would cause inaccuracies at the burst edges.
    # Therefore, we first compute KDE on the full timestamps array and then
    # we take slices for each burst.
    # These KDEs are evaluated on all-photons array `ph` (hence the Ti suffix)
    # using D or A photons during D-excitation (argument ph[mask_d] or ph[mask_a]).
    KDE_DTi = phrates.kde_laplace(ph[mask_d], tau, time_axis=ph)
    KDE_ATi = phrates.kde_laplace(ph[mask_a], tau, time_axis=ph)

    FRET_2CDE = []
    for ib, burst in enumerate(bursts):
        burst_slice = slice(int(burst.istart), int(burst.istop) + 1)
        if ~mask_d[burst_slice].any() or ~mask_a[burst_slice].any():
            # Either D or A photon stream has no photons in current burst,
            # thus FRET_2CDE cannot be computed. Fill position with NaN.
            print("nan")
            FRET_2CDE.append(np.nan)
            continue

        # Take slices of KDEs for current burst
        kde_adi = KDE_ATi[burst_slice][mask_d[burst_slice]]
        kde_ddi = KDE_DTi[burst_slice][mask_d[burst_slice]]
        kde_dai = KDE_DTi[burst_slice][mask_a[burst_slice]]
        kde_aai = KDE_ATi[burst_slice][mask_a[burst_slice]]

        # nbKDE does not include the "center" timestamp which contributes 1.
        # We thus subtract 1 from the precomputed KDEs.
        # The N_CHD (N_CHA) value in the correction factor is the number of
        # timestamps in DexDem (DexAem) stream falling within the current burst.
        N_CHD = mask_d[burst_slice].sum()
        N_CHA = mask_a[burst_slice].sum()
        nbkde_ddi = (1 + 2 / N_CHD) * (kde_ddi - 1)
        nbkde_aai = (1 + 2 / N_CHA) * (kde_aai - 1)

        # N_CHD (N_CHA) in eq. 6 (eq. 7) of (Tomov 2012) is the number of photons
        # in DexDem (DexAem) in current burst. Thus the sum is a mean.
        ED = np.mean(kde_adi / (kde_adi + nbkde_ddi))  # (E)_D
        if np.isnan(ED):
            print(kde_adi, nbkde_ddi)
            print(N_CHD)
            print("______")
        EA = np.mean(kde_dai / (kde_dai + nbkde_aai))  # (1 - E)_A

        # Compute fret_2cde for current burst
        fret_2cde = 110 - 100 * (ED + EA)
        FRET_2CDE.append(fret_2cde)
    return np.array(FRET_2CDE)


fret_2cde = calc_fret_2cde(tau, ph, mask_d, mask_a, bursts)
fret_2cde
# %%

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

    # !! changed to + compared to fretbrusts code
    alex_2cde = 100 - 50 * (BRDex[-1] + BRAex[-1])
    ALEX_2CDE.append(alex_2cde)
ALEX_2CDE = np.array(ALEX_2CDE)


# %%
timestamps = ph
streams = {
    "DD": {"Dex": "Dem"},
    "DA": {"Dex": "Aem"},
    "AA": {"Aex": "Aem"},
    "AD": {"Aex": "Dem"},
}

stream = np.empty_like(timestamps, dtype="U2")
for stream_label, kwargs in streams.items():
    mask = d.get_ph_mask(ph_sel=Ph_sel(**kwargs))
    stream[mask] = stream_label

# %%


photons_export = pl.DataFrame({"timestamps": timestamps, "stream": stream})
photons_export.write_parquet("photon_data.pq")

# %%


burst_export = pl.DataFrame([{"istart": b.istart, "istop": b.istop} for b in bursts])
burst_export = burst_export.with_columns(
    [
        pl.lit(ALEX_2CDE).alias("alex_2cde"),
        pl.lit(fret_2cde).alias("fret_2cde"),
    ]
)
burst_export.write_parquet("burst_data.pq")
