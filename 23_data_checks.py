"""
23_data_checks.py
Revisi putaran 2 (Reviewer 2): pemeriksaan data dan struktur galat.

  HA-3  ADF dan KPSS pada log CPI (tren deterministik) dan pada inflasi bulanan r_t
  DG-2  r1 selisih pertama dan kedua dengan CI bootstrap blok, estimasi rho lewat
        inversi AR(1), klasifikasi langkah (iii), cek konsistensi SNR terhadap SNR0(rho_hat)
  DA-3  variansi residual r_t per rezim cakupan kota (66, 82, 90, 150 kota)
  DA-2  apakah residual Test (A) mengumpul di sekitar bulan penautan tahun dasar
        (Januari 2014, Januari 2020, Januari 2024); uji pergeseran sirkular

Varian musiman S1 (12 dummy bulan, kontras jumlah-nol) dan S2 (S1 + indikator bulan
Idulfitri dan bulan sebelumnya) dibangun ulang di sini; kecocokannya dengan naskah
dicek lewat r1 selisih kedua log indeks (-0.166, -0.223) dan SNR inflasi (0.331, 0.365).

Keluaran: output/dc_unitroot.csv, output/dc_r1_rho.csv, output/dc_noise_floor_matched.csv,
          output/dc_regimes.csv, output/dc_testA_repro.csv, output/dc_links.csv
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from lpr_core import ar1_corr, weight_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SEED, B_BOOT, BLOCK = 20261007, 2000, 12
pd.set_option("display.width", 200)

df = pd.read_csv(OUT / "ihk_bps_2010_2026.csv")
n = len(df)
T = np.arange(1, n + 1, dtype=float)
X = T / n
r = df["infl_pct_mtm"].to_numpy(float)
month, year = df["month"].to_numpy(), df["year"].to_numpy()

# ---------- varian musiman S1, S2 ----------
EID = {2010: 9, 2011: 8, 2012: 8, 2013: 8, 2014: 7, 2015: 7, 2016: 7, 2017: 6, 2018: 6,
       2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3, 2026: 3}


def seasonal_adjust(rr, with_eid):
    cols = [np.ones(n)] + [(month == j).astype(float) - (month == 12).astype(float)
                           for j in range(1, 12)]
    if with_eid:
        eid = np.array([month[i] == EID[year[i]] for i in range(n)], float)
        cols += [eid, np.roll(eid, -1)]          # bulan Idulfitri dan bulan sebelumnya
    Z = np.column_stack(cols)
    beta = np.linalg.lstsq(Z, rr, rcond=None)[0]
    return rr - Z[:, 1:] @ beta[1:]


def log_index(rr):
    return np.log(100.0) + np.cumsum(np.log1p(rr / 100.0))


VAR_R = {"raw": r, "S1": seasonal_adjust(r, False), "S2": seasonal_adjust(r, True)}
VAR_L = {k: log_index(v) for k, v in VAR_R.items()}

# ---------- alat bantu ----------
L12 = weight_matrix(X, X, 12 / n, 1)
I_n = np.eye(n)
C = I_n - np.ones((n, n)) / n


def snr(y):
    fit = L12 @ y
    return fit.std() / (y - fit).std()


def snr0(rho):
    S = I_n if rho == 0 else ar1_corr(n, rho)
    num = np.trace(C @ L12 @ S @ L12.T @ C)
    den = np.trace(C @ (I_n - L12) @ S @ (I_n - L12).T @ C)
    return np.sqrt(num / den)


def acf1(z):
    z = z - z.mean()
    return float((z[1:] * z[:-1]).sum() / (z * z).sum())


def mbb_ci(z, rng):
    m = len(z); k = int(np.ceil(m / BLOCK))
    vals = np.empty(B_BOOT)
    for b in range(B_BOOT):
        st = rng.integers(0, m - BLOCK + 1, k)
        vals[b] = acf1(z[(st[:, None] + np.arange(BLOCK)).ravel()[:m]])
    return np.percentile(vals, [2.5, 97.5])


R1_RHO07 = -(0.7**2 - 3 * 0.7 + 4) / (2 * (3 - 0.7))      # = -0.5196, batas aturan blok


def rho_from_d2(r1):
    """Inversi r1(Delta^2 e) = -(rho^2 - 3rho + 4)/(2(3 - rho)) untuk AR(1), rho in (-1, 1)."""
    if not (-1 < r1 < -0.5):
        return np.nan
    return (3 + 2 * r1 - np.sqrt(4 * r1**2 - 12 * r1 - 7)) / 2


def rho_from_d1(r1):
    """Inversi r1(Delta e) = -(1 - rho)/2 untuk AR(1)."""
    return 1 + 2 * r1 if -1 < r1 < 0 else np.nan


# Aturan langkah (iii) versi naskah revisi 1. Untuk inflasi (stasioner menurut ADF/KPSS, bagian 1)
# aturan ini keliru memberi label "terintegrasi": galatnya stasioner tetapi bukan AR(1) (bagian 2b).
def classify(r1):
    if r1 < -2 / 3:
        return "r1 < -2/3: ketergantungan negatif"
    if r1 <= R1_RHO07:
        return "AR(1), rho <= 0.7: aturan blok"
    if r1 < -0.5:
        return "AR(1), rho > 0.7: di luar validasi"
    if r1 < 0:
        return "terintegrasi/persisten: Pers. (26)"
    return "r1 >= 0: siklik/musiman, sesuaikan dulu"


# ---------- (1) uji akar unit dan stasioneritas ----------
def adf(x, reg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        st, p, lag, nobs, cv, _ = adfuller(x, regression=reg, autolag="AIC")
    return {"test": "ADF", "reg": reg, "stat": st, "p": p, "lags": lag, "nobs": nobs, "cv5": cv["5%"]}


def kpss_(x, reg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        st, p, lag, cv = kpss(x, regression=reg, nlags="auto")
    return {"test": "KPSS", "reg": reg, "stat": st, "p": p, "lags": lag, "nobs": len(x), "cv5": cv["5%"]}


ur = []
for v in ["raw", "S2"]:
    for f in (adf(VAR_L[v], "ct"), adf(VAR_L[v], "ctt"), kpss_(VAR_L[v], "ct")):
        ur.append({"series": "log CPI", "variant": v, **f})
    for f in (adf(VAR_R[v], "c"), adf(VAR_R[v], "ct"), kpss_(VAR_R[v], "c"), kpss_(VAR_R[v], "ct")):
        ur.append({"series": "inflation", "variant": v, **f})
ur = pd.DataFrame(ur)
ur["p_note"] = np.where((ur.test == "KPSS") & (ur.p <= 0.01), "p <= 0.01",
                        np.where((ur.test == "KPSS") & (ur.p >= 0.10), "p >= 0.10", ""))
ur.to_csv(OUT / "dc_unitroot.csv", index=False)
print("(1) ADF (H0: akar unit) dan KPSS (H0: stasioner); reg c = konstanta, ct = + tren linear,"
      " ctt = + tren kuadratik")
print(ur.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- (2) r1 selisih, estimasi rho, konsistensi SNR ----------
rng = np.random.default_rng(SEED)
rows = []
for series, VAR in (("log CPI", VAR_L), ("inflation", VAR_R)):
    for v, y in VAR.items():
        d1, d2 = np.diff(y), np.diff(y, 2)
        r11, r12 = acf1(d1), acf1(d2)
        lo1, hi1 = mbb_ci(d1, rng)
        lo2, hi2 = mbb_ci(d2, rng)
        rho2 = rho_from_d2(r12)
        rows.append({"series": series, "variant": v, "r1_D1": r11, "r1_D1_lo": lo1, "r1_D1_hi": hi1,
                     "r1_D2": r12, "r1_D2_lo": lo2, "r1_D2_hi": hi2,
                     "rho_hat_D1": rho_from_d1(r11), "rho_hat_D2": rho2,
                     "class_point": classify(r12),
                     "class_CI": classify(lo2) if classify(lo2) == classify(hi2) else "CI melintasi batas",
                     "SNR": snr(y),
                     "SNR0_at_rho_hat_D2": snr0(rho2) if np.isfinite(rho2) else np.nan})
r1tab = pd.DataFrame(rows)
r1tab.to_csv(OUT / "dc_r1_rho.csv", index=False)
print(f"\n(2) r1 selisih pertama (D1) dan kedua (D2), CI 95% bootstrap blok (l = {BLOCK}, B = {B_BOOT});"
      f" batas rho = 0.7 pada r1(D2) = {R1_RHO07:.4f}")
print(r1tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("Noise floor SNR0 (LL, jendela 12 titik) untuk AR(1): "
      + ", ".join(f"rho={p}: {snr0(p):.3f}" for p in (0.0, 0.3, 0.55, 0.6, 0.67, 0.75)))
for v in ("raw", "S2"):
    e = VAR_R[v] - L12 @ VAR_R[v]
    ac = [acf1(e) if k == 1 else float(np.corrcoef(e[k:], e[:-k])[0, 1]) for k in range(1, 14)]
    print(f"ACF residual LL (h = 12) inflasi {v}, lag 1-13: " + " ".join(f"{a:+.2f}" for a in ac))

# ---------- (2b) noise floor dengan dependensi yang ditaksir tanpa smoother pilot ----------
# Variogram V(k) = E[(Y_{t+k} - Y_t)^2]/2 = gamma(0) - gamma(k) bila tren mulus dapat diabaikan
# dalam k bulan; gamma(0) dari dataran V(k), k = 25..36. Matriks korelasi Toeplitz ditaper
# Bartlett hingga lag 24 agar semi-definit positif, lalu dimasukkan ke Persamaan (22).
def variogram_rho(y, kmax=36, plateau=(25, 36), lag_max=24):
    V = np.array([0.5 * np.mean((y[k:] - y[:-k]) ** 2) for k in range(1, kmax + 1)])
    g0 = V[plateau[0] - 1:plateau[1]].mean()
    rho = 1 - V[:lag_max] / g0
    taper = 1 - np.arange(1, lag_max + 1) / (lag_max + 1)
    return rho, rho * taper


def snr0_matrix(S):
    num = np.trace(C @ L12 @ S @ L12.T @ C)
    den = np.trace(C @ (I_n - L12) @ S @ (I_n - L12).T @ C)
    return np.sqrt(num / den)


print("\n(2b) Dependensi galat inflasi dari variogram (tanpa pilot) dan noise floor padanannya")
vg_rows = []
for v in ("raw", "S1", "S2"):
    rho_raw, rho_tap = variogram_rho(VAR_R[v])
    col = np.r_[1.0, rho_tap, np.zeros(n - 1 - len(rho_tap))]
    S = col[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    mineig = np.linalg.eigvalsh(S).min()
    vg_rows.append({"variant": v, "rho1": rho_raw[0], "rho2": rho_raw[1], "rho3": rho_raw[2],
                    "rho6": rho_raw[5], "rho12": rho_raw[11], "min_eig": mineig,
                    "SNR": snr(VAR_R[v]), "SNR0_matched": snr0_matrix(S)})
    print(f"  {v}: rho_k (k=1..13) = " + " ".join(f"{x:+.2f}" for x in rho_raw[:13]))
vg = pd.DataFrame(vg_rows)
vg.to_csv(OUT / "dc_noise_floor_matched.csv", index=False)
print(vg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------- (3) rezim cakupan kota ----------
REG = [(1, 48, "2010-13 (66 kota)"), (49, 120, "2014-19 (82 kota)"),
       (121, 168, "2020-23 (90 kota)"), (169, 198, "2024-26 (150 kota)")]
OUTL = np.abs(r) > 1.5                        # empat outlier yang disebut naskah
reg_rows, pvals = [], {}
for v in ("raw", "S2"):
    e = VAR_R[v] - L12 @ VAR_R[v]
    groups, groups_no = [], []
    for a, b, lab in REG:
        sl = slice(a - 1, b)
        groups.append(e[sl]); groups_no.append(e[sl][~OUTL[sl]])
        reg_rows.append({"variant": v, "regime": lab, "n": b - a + 1, "mean_r": VAR_R[v][sl].mean(),
                         "sd_r": VAR_R[v][sl].std(ddof=1), "sd_resid": e[sl].std(ddof=1),
                         "sd_resid_no_outlier": e[sl][~OUTL[sl]].std(ddof=1)})
    pvals[v] = {"BF_all4": stats.levene(*groups, center="median").pvalue,
                "BF_all4_no_outlier": stats.levene(*groups_no, center="median").pvalue,
                **{f"BF_{REG[k][2][:7]}_vs_{REG[k + 1][2][:7]}":
                   stats.levene(groups[k], groups[k + 1], center="median").pvalue for k in range(3)}}
regt = pd.DataFrame(reg_rows)
regt.to_csv(OUT / "dc_regimes.csv", index=False)
print("\n(3) Residual r_t dari fit LL h = 12 per rezim cakupan; uji Brown-Forsythe (Levene, median)")
print(regt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for v, d in pvals.items():
    print(f"  p-value {v}: " + ", ".join(f"{k} = {p:.3f}" for k, p in d.items()))

# ---------- (4) Test (A): reproduksi Tabel 5 dan residual di sekitar bulan penautan ----------
def coef_rows(t0, Ts, h, p):
    u = (Ts - t0) / h
    w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    Z = np.vander(Ts - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


LINKS = (49, 121, 169)                      # Jan 2014, Jan 2020, Jan 2024
rep_rows, link_rows = [], []
for series, y in (("log CPI", np.log(df["IHK_recon"].to_numpy(float))), ("inflation", r)):
    for h in (9, 12, 18, 24):
        t0s = np.arange(48, n - h + 1)
        k = np.arange(h); Kk = 0.75 * (1 - (k / h) ** 2)
        delta = (k * Kk).sum() / Kk.sum()
        rev = {0: [], 1: []}; m1, m2 = [], []
        gam = None
        for t0 in t0s:
            for p in (0, 1):
                fin = coef_rows(t0, T, h, p)[0]
                rt = coef_rows(t0, T[:t0], h, p)[0]
                rev[p].append(fin @ y - rt @ y[:t0])
                if p == 1 and gam is None:
                    gam = 0.5 * ((fin * (T - t0) ** 2).sum() - (rt * (T[:t0] - t0) ** 2).sum())
            m1.append(coef_rows(t0, T, h, 1)[1] @ y)
            m2.append(2 * coef_rows(t0, T, h, 2)[2] @ y)
        for est, regr, bench in (("NW", np.array(m1), delta), ("LL", np.array(m2), gam)):
            dep = np.array(rev[0 if est == "NW" else 1])
            b, a = np.polyfit(regr, dep, 1)
            res = dep - (a + b * regr)
            rep_rows.append({"series": series, "h": h, "est": est, "n_t0": len(t0s), "slope": b,
                             "benchmark": bench, "ratio": b / bench,
                             "corr": np.corrcoef(regr, dep)[0, 1]})
            near = np.zeros(len(t0s), bool)
            for L in LINKS:
                near |= np.abs(t0s - L) <= h - 1
            e2 = res ** 2
            stat0 = e2[near].mean() / e2[~near].mean()
            shifts = np.array([np.roll(e2, s)[near].mean() / np.roll(e2, s)[~near].mean()
                               for s in range(len(e2))])
            link_rows.append({"series": series, "h": h, "est": est, "n_t0": len(t0s),
                              "n_near": int(near.sum()), "ms_ratio_near_far": stat0,
                              "p_circular_shift": (shifts >= stat0).mean()})
rep = pd.DataFrame(rep_rows); links = pd.DataFrame(link_rows)
rep.to_csv(OUT / "dc_testA_repro.csv", index=False)
links.to_csv(OUT / "dc_links.csv", index=False)
print("\n(4a) Reproduksi Test (A) (NW: revisi vs slope LL, benchmark delta_h; LL: revisi vs kelengkungan LQ,"
      " benchmark gamma_h)")
print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n(4b) Residual Test (A) di sekitar bulan penautan (|t0 - L| <= h - 1, L = Jan 2014/2020/2024):"
      " rasio rata-rata kuadrat dekat/jauh, p-value pergeseran sirkular")
print(links.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
