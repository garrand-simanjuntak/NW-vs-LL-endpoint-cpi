"""
24_worked_example.py
Revisi putaran 2 (Reviewer 2), K6: contoh kerja prosedur diagnostik Bagian 3.7 pada kedua respons
(DG-1, DG-2) dan implikasi kebijakan (DG-4).

Prasyarat, keluaran skrip lain di output/: 14_snr_sweep.py (snr_crossover.csv), 21_testA_eiv.py
(k2_data.csv), 22_testB_sensitivity.py (k3_panelB.csv), 23_data_checks.py (dc_unitroot.csv,
dc_r1_rho.csv, dc_noise_floor_matched.csv).

Langkah:
 (i)   Validitas instrumen pada h = 12: rasio slope Test (A) terhadap benchmark yang disesuaikan noise
       (Pers. 25a), dibaca dari K2.
 (ii)  Biaya jendela tetap: lag rata-rata bobot NW satu sisi delta_h (h = 12) dan rata-rata trailing
       12 bulan (5.5 bulan); revisi NW rata-rata pada log CPI dalam poin log.
 (iii) Struktur galat. ADF dan KPSS dengan konstanta dan tren linear (dari K5), aturan empat hasil pada
       5%: ADF tolak dan KPSS tidak -> stasioner; ADF tidak dan KPSS tolak -> terintegrasi; keduanya tolak
       atau keduanya tidak -> persisten. Terintegrasi dan persisten -> Pers. (26). Bila stasioner, cek
       kecukupan AR(1) tiga bagian dengan rho_D1 = 1 + 2 r1(Delta y) (Pers. 20a): (a) nilai AR(1)
       r(rho) = -(rho^2 - 3 rho + 4)/(2(3 - rho)) di dalam CI 95% bootstrap blok r1(Delta^2 y) (K5);
       (b) korelasi lag 1 variogram di dalam pita 95% di bawah AR(1)(rho_D1) (dua estimator rho tanpa pilot
       harus sama di bawah AR(1)); (c) SNR tidak di bawah SNR noise AR(1)(rho_D1) murni,
       p = P(SNR simulasi <= SNR teramati). (b) dan (c) dari 2000 deret noise AR(1) murni. AR(1) memadai
       hanya bila ketiganya lolos. Bila tidak: lambda umum
       lambda(l) = 2 sum_{j di luar blok} w_j Sigma_ij / (w' Sigma w), bobot LL leave-block-out eksak, Sigma
       variogram ber-taper sampai lag 24 pada S2 (cek lag 12 dulu), presisinya pada n = 198 (300 replikasi
       Gaussian dengan Sigma S2), dan kerataan kriteria MCV. Bandwidth akhirnya dari Pers. (26) (K3).
 (iv)  SNR terhadap noise floor i.i.d. dan floor padanan (K5), posisi pada pita crossover (K4).
 (v)   DG-4: tren inflasi tahunan (12 x tren bulanan) real-time NW h = 12, LL h = 12 dan rata-rata trailing
       12 bulan terhadap estimasi final LL interior h = 12, pada S2, episode 2013-14, 2020, 2022.
 (vi)  Gambar 5 (diagram alir): sumber Graphviz selalu ditulis; PNG dirender bila program dot tersedia.

Keluaran: output/k6_steps.csv, output/k6_ar1_check.csv, output/k6_lambda.csv, output/k6_mcv.csv,
          output/k6_dg4.csv, output/fig5_flowchart.dot, output/fig5_flowchart.png
"""
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from lpr_core import weight_matrix, ar1_corr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
t_start = time.time()
NEED = ["snr_crossover.csv", "k2_data.csv", "k3_panelB.csv", "dc_unitroot.csv", "dc_r1_rho.csv",
        "dc_noise_floor_matched.csv"]
missing = [f for f in NEED if not (OUT / f).exists()]
if missing:
    raise SystemExit(f"Jalankan 14, 21, 22 dan 23 lebih dulu; tidak ditemukan: {missing}")

df = pd.read_csv(OUT / "ihk_bps_2010_2026.csv", parse_dates=["date"])
n = len(df)
T = np.arange(1, n + 1, dtype=float)
X = T / n
H = 12
ALPHA = 0.05
SEED, R_SNR, R_LAM = 20261011, 2000, 300
EID_MONTH = {2010: 9, 2011: 8, 2012: 8, 2013: 8, 2014: 7, 2015: 7, 2016: 7, 2017: 6,
             2018: 6, 2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3, 2026: 3}
f3 = lambda v: f"{v:.3f}"


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def lp0(t0, Tv, h, p):
    w = epan((Tv - t0) / h)
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]


def adjust_s2(r):
    """Identik dengan 16_testB_own_bandwidth.py dan 22_testB_sensitivity.py."""
    month = df.date.dt.month.values
    D = np.column_stack([(month == m).astype(float) for m in range(1, 13)])
    eid = np.array([1.0 if m == EID_MONTH[y] else 0.0 for y, m in zip(df.date.dt.year, month)])
    pre = np.roll(eid, -1); pre[-1] = 0.0
    S = np.column_stack([D, eid, pre]); S = S - S.mean(axis=0)
    Xd = np.column_stack([np.ones(n), S])
    b = np.linalg.lstsq(Xd, r, rcond=None)[0]
    return r - S @ b[1:]


def variogram_cov(y, kmax=36, plateau=(25, 36), lag_max=24):
    """Sama dengan 21, 22 dan 23."""
    V = np.array([0.5 * np.mean((y[k:] - y[:-k]) ** 2) for k in range(1, kmax + 1)])
    g0 = V[plateau[0] - 1:plateau[1]].mean()
    rho = 1 - V[:lag_max] / g0
    return g0, rho, rho * (1 - np.arange(1, lag_max + 1) / (lag_max + 1))


def toeplitz(rho_tap, size):
    col = np.r_[1.0, rho_tap, np.zeros(size - 1 - len(rho_tap))]
    return col[np.abs(np.subtract.outer(np.arange(size), np.arange(size)))]


def r_ar1(rho):
    """Autokorelasi lag 1 selisih kedua galat AR(1)."""
    return -(rho**2 - 3 * rho + 4) / (2 * (3 - rho))


def ell_star(rho):
    """Pers. (20): min{l >= 0: rho^l <= 0.1}, = 1 untuk rho <= 0.1."""
    r = min(max(float(rho), 0.0), 0.99)
    return 1 if r <= 0.1 else int(np.ceil(np.log(0.1) / np.log(r) - 1e-12))


L12 = weight_matrix(X, X, H / n, 1)
I_n = np.eye(n)
C_n = I_n - 1.0 / n


def snr(Y):
    F = L12 @ Y
    return F.std(axis=0) / (Y - F).std(axis=0)


def snr0(S):
    return np.sqrt(np.trace(C_n @ L12 @ S @ L12.T @ C_n) / np.trace(C_n @ (I_n - L12) @ S @ (I_n - L12).T @ C_n))


r_raw = df.infl_pct_mtm.to_numpy(float)
r_s2 = adjust_s2(r_raw)
logcpi = np.log(100 * np.cumprod(1 + r_raw / 100))
rng = np.random.default_rng(SEED)
rows = []

# ---------- (i) validitas instrumen ----------
k2 = pd.read_csv(OUT / "k2_data.csv")
k2 = k2[(k2.h == H) & (k2.H == H)]
print(f"(i) TEST (A) PADA h = {H}: rasio slope, CI HAC, benchmark disesuaikan noise (Pers. 25a)")
for r in k2.itertuples():
    print(f"  {r.series:9s} {r.est}: rasio {r.ratio:.3f} [{r.HAC_lo:.3f}, {r.HAC_hi:.3f}], benchmark "
          f"{r.ratio_null:.3f}, porsi noise regresor {min(r.noise_share, 1):.2f}, valid = {r.null_in_HAC}")
    rows.append({"step": "i", "series": r.series, "item": f"Test A {r.est} ratio", "value": r.ratio,
                 "lo": r.HAC_lo, "hi": r.HAC_hi, "ref": r.ratio_null, "verdict": "valid" if r.null_in_HAC else "invalid"})

# ---------- (ii) biaya jendela tetap ----------
k = np.arange(H)
delta_h = float((k * (1 - (k / H) ** 2)).sum() / (1 - (k / H) ** 2).sum())
lag_tm = (H - 1) / 2
t0s = np.arange(48, n - H + 1)
fin = np.array([lp0(t0, T, H, 1) @ logcpi for t0 in t0s])
rt_nw = np.array([lp0(t0, T[:t0], H, 0) @ logcpi[:t0] for t0 in t0s])
rev = fin - rt_nw
slope = np.array([np.polyfit(T[t0 - H:t0 + H - 1] - t0, logcpi[t0 - H:t0 + H - 1], 1)[0] for t0 in t0s])
print(f"\n(ii) LAG JENDELA TETAP h = {H}: delta_h = {delta_h:.4f} bulan (NW satu sisi), rata-rata trailing "
      f"{H} bulan = {lag_tm:.1f} bulan")
w_nw = 1 - (k / H) ** 2; w_nw = w_nw / w_nw.sum()
w_ll = lp0(float(n), T, H, 1)[n - H:][::-1]                      # bobot LL di titik ujung, lag 0..h-1
v_nw, v_tm, v_ll = (w_nw**2).sum(), 1.0 / H, (w_ll**2).sum()
print(f"  faktor varians (galat i.i.d.) sum w^2: NW {v_nw:.4f}, rata-rata trailing {v_tm:.4f}, LL {v_ll:.4f}; "
      f"NW/trailing = {v_nw / v_tm:.3f}, LL/trailing = {v_ll / v_tm:.3f}; lag rata-rata bobot LL "
      f"{(np.arange(H) * w_ll).sum():+.4f} bulan")
rows += [{"step": "ii", "series": "both", "item": "variance factor NW/trailing", "value": v_nw / v_tm},
         {"step": "ii", "series": "both", "item": "variance factor LL/trailing", "value": v_ll / v_tm}]
print(f"  log CPI: revisi NW rata-rata {rev.mean():.5f} poin log (median {np.median(rev):.5f}), "
      f"positif pada {np.mean(rev > 0):.2f} dari t0; slope lokal rata-rata {slope.mean():.5f} per bulan, "
      f"delta_h x slope = {delta_h * slope.mean():.5f}")
rows += [{"step": "ii", "series": "both", "item": "delta_h (months)", "value": delta_h},
         {"step": "ii", "series": "both", "item": "trailing-mean lag (months)", "value": lag_tm},
         {"step": "ii", "series": "log CPI", "item": "mean NW revision (log points)", "value": rev.mean()},
         {"step": "ii", "series": "log CPI", "item": "share of t0 with positive revision", "value": np.mean(rev > 0)}]

# ---------- (iii) struktur galat ----------
ur = pd.read_csv(OUT / "dc_unitroot.csv")


def ur_p(series, variant, test, reg):
    q = ur[(ur.series == series) & (ur.variant == variant) & (ur.test == test) & (ur.reg == reg)]
    return float(q.p.iloc[0])


def classify(p_adf, p_kpss):
    adf_rej, kpss_rej = p_adf < ALPHA, p_kpss <= ALPHA
    if adf_rej and not kpss_rej:
        return "stationary"
    if kpss_rej and not adf_rej:
        return "integrated"
    return "persistent (conflicting)"


print("\n(iii) STRUKTUR GALAT. ADF dan KPSS dengan konstanta dan tren linear, 5%; KPSS p dipotong ke [0.01, 0.10]")
CLASS = {}
for series in ("log CPI", "inflation"):
    for v in ("raw", "S2"):
        pa, pk = ur_p(series, v, "ADF", "ct"), ur_p(series, v, "KPSS", "ct")
        CLASS[(series, v)] = classify(pa, pk)
        extra = ""
        if series == "inflation":
            pa_c, pk_c = ur_p(series, v, "ADF", "c"), ur_p(series, v, "KPSS", "c")
            extra = f" | dengan konstanta saja: ADF p = {pa_c:.3f}, KPSS p = {pk_c:.2f} -> {classify(pa_c, pk_c)}"
        else:
            extra = f" | ADF dengan tren kuadratik p = {ur_p(series, v, 'ADF', 'ctt'):.3f}"
        print(f"  {series:9s} {v:3s}: ADF p = {pa:.3f}, KPSS p = {pk:.2f} -> {CLASS[(series, v)]}{extra}")
        rows.append({"step": "iii", "series": series, "item": f"{v} ADF p (ct)", "value": pa})
        rows.append({"step": "iii", "series": series, "item": f"{v} KPSS p (ct)", "value": pk,
                     "verdict": CLASS[(series, v)]})

r1 = pd.read_csv(OUT / "dc_r1_rho.csv")
nf = pd.read_csv(OUT / "dc_noise_floor_matched.csv").set_index("variant")


def vario_rho1(Y, plateau=(25, 36)):
    """Korelasi lag 1 dari variogram, per kolom (sama dengan variogram_cov)."""
    V1 = 0.5 * np.mean((Y[1:] - Y[:-1]) ** 2, axis=0)
    g0 = np.mean([0.5 * np.mean((Y[k:] - Y[:-k]) ** 2, axis=0) for k in range(plateau[0], plateau[1] + 1)], axis=0)
    return 1 - V1 / g0


def d1_mean(Y):
    """Rata-rata rho_D1 = 1 + 2 r1(Delta y) atas kolom (cek bahwa simulasi mereproduksi rho)."""
    d = np.diff(Y, axis=0); d = d - d.mean(axis=0)
    return float(np.mean(1 + 2 * (d[1:] * d[:-1]).sum(axis=0) / (d * d).sum(axis=0)))


ar_rows = []
print("  Kecukupan AR(1) untuk deret stasioner, rho_D1 dari Pers. (20a). (a) r(rho_D1) di dalam CI bootstrap "
      "r1(Delta^2 y) (K5); (b) korelasi lag 1 variogram di dalam pita 95% di bawah AR(1)(rho_D1); (c) SNR tidak di "
      f"bawah noise AR(1)(rho_D1) murni. (b) dan (c) dari {R_SNR} deret noise AR(1) murni, n = {n}")
for v, y in (("raw", r_raw), ("S2", r_s2)):
    if CLASS[("inflation", v)] != "stationary":
        continue
    q = r1[(r1.series == "inflation") & (r1.variant == v)].iloc[0]
    rho = float(q.rho_hat_D1)
    ra = r_ar1(rho)
    ok_a = bool(q.r1_D2_lo <= ra <= q.r1_D2_hi)
    z = rng.standard_normal((n, R_SNR))
    e = np.empty_like(z)
    e[0] = z[0]                                   # start stasioner
    e[1:] = lfilter([np.sqrt(1 - rho**2)], [1, -rho], z[1:], axis=0, zi=rho * z[0][None, :])[0]
    v_obs = float(vario_rho1(y[:, None])[0])
    v_sim = vario_rho1(e)
    v_lo, v_hi = np.percentile(v_sim, [2.5, 97.5])
    ok_b = bool(v_lo <= v_obs <= v_hi)
    p_b = float(np.mean(v_sim <= v_obs))
    s_obs = float(snr(y[:, None])[0])
    s0 = float(snr0(ar1_corr(n, rho)))
    s_sim = snr(e)
    p_c = float(np.mean(s_sim <= s_obs))
    ok_c = p_c >= ALPHA
    ar1_ok = bool(ok_a and ok_b and ok_c)
    rho_v = nf.loc[v, ["rho1", "rho2"]].to_numpy(float)
    ar_rows.append({"variant": v, "rho_D1": rho, "r1_D2": q.r1_D2, "r1_D2_lo": q.r1_D2_lo, "r1_D2_hi": q.r1_D2_hi,
                    "r_ar1_at_rho": ra, "a_pass": ok_a, "vario_rho1": v_obs, "vario_band_lo": v_lo,
                    "vario_band_hi": v_hi, "p_b": p_b, "D1_sim_mean": d1_mean(e), "b_pass": ok_b,
                    "SNR": s_obs, "SNR0_rho": s0, "SNR_sim_p05": np.percentile(s_sim, 5), "SNR_sim_min": s_sim.min(),
                    "p_c": p_c, "c_pass": ok_c, "AR1_adequate": ar1_ok,
                    "ell_star_rho": ell_star(rho), "vario_rho2": rho_v[1], "rho1_sq": rho_v[0] ** 2,
                    "D1_implied": (rho_v[0] - rho_v[1]) / (1 - rho_v[0])})
    print(f"  inflasi {v:3s}: rho_D1 = {rho:.3f}, l*(rho_D1) = {ell_star(rho)}")
    print(f"      (a) r(rho) = {ra:.4f}; r1(D2) = {q.r1_D2:.4f} [{q.r1_D2_lo:.4f}, {q.r1_D2_hi:.4f}] -> "
          f"{'lolos' if ok_a else 'gagal'}")
    print(f"      (b) variogram rho1 = {v_obs:.3f}; pita 95% di bawah AR(1)({rho:.3f}) = [{v_lo:.3f}, {v_hi:.3f}], "
          f"median {np.median(v_sim):.3f}, P(sim <= data) = {p_b:.4f} -> {'lolos' if ok_b else 'gagal'}; variogram rho2 = {rho_v[1]:+.3f} "
          f"(AR(1): rho1^2 = {rho_v[0] ** 2:.3f}), (rho1 - rho2)/(1 - rho1) = {(rho_v[0] - rho_v[1]) / (1 - rho_v[0]):.3f}")
    print(f"      (c) SNR = {s_obs:.3f}; SNR0(rho) = {s0:.3f}; noise murni: persentil 5% {np.percentile(s_sim, 5):.3f}, "
          f"minimum {s_sim.min():.3f}; p = {p_c:.4f} -> {'lolos' if ok_c else 'gagal'}")
    print(f"      AR(1) memadai: {ar1_ok}")
ar = pd.DataFrame(ar_rows); ar.to_csv(OUT / "k6_ar1_check.csv", index=False)

# lambda umum (Pers. 20b) pada S2, setelah cek lag 12
print("\n  Lambda umum. Cek lag 12 dulu (variogram, lag 12 dan 24):")
VARS = {"raw": r_raw, "S2": r_s2}
for v, y in VARS.items():
    g0, rho, _ = variogram_cov(y)
    print(f"    {v:3s}: rho12 = {rho[11]:+.3f}, rho24 = {rho[23]:+.3f}, rho6 = {rho[5]:+.3f}")
g0_s2, rho_s2, tap_s2 = variogram_cov(r_s2)
SIG_s2 = toeplitz(tap_s2, n)


def lbo_rows(h, ell):
    """Bobot LL leave-block-out (titik |j - i| <= ell dihapus) untuk semua i; None bila tak layak."""
    D = X[None, :] - X[:, None]
    U = D / h
    K = np.where(np.abs(U) <= 1, 0.75 * (1 - U**2), 0.0)
    K = K * (np.abs(np.arange(n)[None, :] - np.arange(n)[:, None]) > ell)
    if ((K > 0).sum(axis=1) < 3).any():
        return None
    S0, S1, S2 = K.sum(1), (K * D).sum(1), (K * D * D).sum(1)
    return K * (S2[:, None] - S1[:, None] * D) / (S0 * S2 - S1**2)[:, None]


def lam_exact(Sig, h, ell, i=n // 2):
    M = lbo_rows(h, ell)
    if M is None:
        return np.nan
    w = M[i]
    return 2 * (w @ Sig[i]) / (w @ Sig @ w)


def stable_ell(lams, tol=0.2, lmin=1):
    ok = np.abs(lams) <= tol
    for l in range(lmin, len(lams)):
        if np.isfinite(lams[l]) and ok[l:][np.isfinite(lams[l:])].all():
            return l
    return -1


H_LAM = [24, 36, 48, 60]
lam_rows = []
for hm in H_LAM:
    lams = np.array([lam_exact(SIG_s2, hm / n, l) for l in range(25)])
    first = next((l for l in range(1, 25) if np.isfinite(lams[l]) and abs(lams[l]) <= 0.2), -1)
    lam_rows.append({"h_months": hm, "first_ell": first, "stable_ell": stable_ell(lams),
                     **{f"lam_{l}": lams[l] for l in range(25)}})
    print(f"    S2, h = {hm} bulan: lambda(l), l = 0..12: " + " ".join(f"{x:+.2f}" for x in lams[:13])
          + f" | pertama |lambda| <= 0.2 (l >= 1): {first}; stabil sampai lag 24: {stable_ell(lams)}")
lam = pd.DataFrame(lam_rows)
# presisi lambda topi pada n = 198
Lc = np.linalg.cholesky(g0_s2 * SIG_s2)
prec = {hm: np.empty((R_LAM, 25)) for hm in (36, 60)}
sel = {hm: np.empty(R_LAM, int) for hm in (36, 60)}
nonpsd = 0
for b in range(R_LAM):
    eb = Lc @ rng.standard_normal(n)
    _, _, tb = variogram_cov(eb)
    Sb = toeplitz(tb, n)
    nonpsd += np.linalg.eigvalsh(Sb).min() < 0
    for hm in (36, 60):
        lb = np.array([lam_exact(Sb, hm / n, l) for l in range(25)])
        prec[hm][b] = lb
        sel[hm][b] = stable_ell(lb)
for hm in (36, 60):
    q1, q3 = np.nanpercentile(prec[hm], [25, 75], axis=0)
    iqr = q3 - q1
    print(f"    presisi (Sigma S2, {R_LAM} deret, h = {hm}): IQR lambda_topi untuk l = 0..8 = "
          f"{iqr[:9].min():.2f}-{iqr[:9].max():.2f}; l stabil terpilih median {np.median(sel[hm]):.0f} "
          f"(kuartil {np.percentile(sel[hm], 25):.0f}-{np.percentile(sel[hm], 75):.0f})")
    lam.loc[lam.h_months == hm, "iqr_l0_8_min"] = iqr[:9].min()
    lam.loc[lam.h_months == hm, "iqr_l0_8_max"] = iqr[:9].max()
    lam.loc[lam.h_months == hm, "stable_ell_sim_median"] = np.median(sel[hm])
print(f"    Sigma topi tidak semidefinit positif pada {nonpsd / R_LAM:.2f} replikasi")
lam.to_csv(OUT / "k6_lambda.csv", index=False)

# kerataan kriteria MCV
H_MCV = np.arange(6, 121)
mcv_rows = []
print("  Kriteria MCV pada inflasi (LL, h = 6..120 bulan): h terpilih, rentang h dengan skor <= 1.01 x minimum")
for v, y in VARS.items():
    l_rule = ell_star(float(ar[ar.variant == v].rho_D1.iloc[0])) if v in set(ar.variant) else None
    for ell in sorted({0, 1, 2, 4, 5, 9, 17} | ({l_rule} if l_rule else set())):
        sc = np.array([np.inf if (M := lbo_rows(hm / n, ell)) is None else np.mean((y - M @ y) ** 2)
                       for hm in H_MCV])
        j = int(np.argmin(sc))
        flat = H_MCV[sc <= 1.01 * sc[j]]
        mcv_rows.append({"variant": v, "ell": ell, "naive_AR1_rule": ell == l_rule, "h_hat": H_MCV[j],
                         "s_hat": (2 * ell + 1) / (2 * H_MCV[j]), "flat1_lo": flat.min(), "flat1_hi": flat.max(),
                         "at_edge": j in (0, len(H_MCV) - 1)})
mcv = pd.DataFrame(mcv_rows); mcv.to_csv(OUT / "k6_mcv.csv", index=False)
print(mcv.to_string(index=False, float_format=f3))

# bandwidth dari Pers. (26)
k3 = pd.read_csv(OUT / "k3_panelB.csv")
print("  Pers. (26), bandwidth masing-masing (K3, h_ref = 12, 18, 24):")
for (v, resp), g in k3.groupby(["variant", "response"], sort=False):
    print(f"    {resp:9s} {v}: h_NW {g.h_NW.min()}-{g.h_NW.max()}, h_LL {g.h_LL.min()}-{g.h_LL.max()}, "
          f"rasio in-sample {g.ratio.min():.3f}-{g.ratio.max():.3f}, OOS {g.ratio_oos.min():.3f}-{g.ratio_oos.max():.3f}")

# ---------- (iv) SNR terhadap noise floor ----------
cr = pd.read_csv(OUT / "snr_crossover.csv")
cr = cr[(cr.n == 198) & (cr.n_cross == 1) & ~cr.target.str.startswith("1x+")]
BAND = {}
for hmax in (1.0, 0.5):
    lo, hi = [], []
    for rho in (0.0, 0.3, 0.6):
        f0 = snr0(I_n if rho == 0 else ar1_corr(n, rho))
        g = cr[(cr.h_max == hmax) & (cr.rho == rho)]
        lo.append(g.SNR_paper_star.min() / f0); hi.append(g.SNR_paper_star.max() / f0)
    BAND[hmax] = (min(lo), max(hi))
f_iid = snr0(I_n)
print(f"\n(iv) SNR / SNR0. Pita crossover dalam kelipatan floor: tanpa batas {BAND[1.0][0]:.2f}-{BAND[1.0][1]:.2f}, "
      f"h <= 0.5 {BAND[0.5][0]:.2f}-{BAND[0.5][1]:.2f}; floor i.i.d. {f_iid:.3f}")


def position(x, band):
    return "below" if x < band[0] else "above" if x > band[1] else "inside"


for v, y in VARS.items():
    s = float(snr(y[:, None])[0])
    m_iid, m_mat = s / f_iid, s / float(nf.loc[v, "SNR0_matched"])
    print(f"  inflasi {v:3s}: SNR = {s:.3f}; /floor i.i.d. = {m_iid:.2f}, /floor padanan "
          f"({nf.loc[v, 'SNR0_matched']:.3f}) = {m_mat:.2f}; tanpa batas: {position(m_iid, BAND[1.0])}, "
          f"h <= 0.5: {position(m_iid, BAND[0.5])}")
    rows += [{"step": "iv", "series": "inflation", "item": f"{v} SNR", "value": s},
             {"step": "iv", "series": "inflation", "item": f"{v} SNR/SNR0 iid", "value": m_iid,
              "verdict": f"uncapped {position(m_iid, BAND[1.0])}, capped {position(m_iid, BAND[0.5])}"},
             {"step": "iv", "series": "inflation", "item": f"{v} SNR/SNR0 matched", "value": m_mat}]
rows += [{"step": "iv", "series": "band", "item": f"h_max {hm} band (x floor)", "lo": BAND[hm][0], "hi": BAND[hm][1]}
         for hm in BAND]
pd.DataFrame(rows).to_csv(OUT / "k6_steps.csv", index=False)

# ---------- (v) DG-4: tren inflasi tahunan pada episode ----------
y = r_s2
t0d = np.arange(13, n - H + 1)
final = np.array([lp0(t0, T, H, 1) @ y for t0 in t0d])
est = {"NW12": np.array([lp0(t0, T[:t0], H, 0) @ y[:t0] for t0 in t0d]),
       "LL12": np.array([lp0(t0, T[:t0], H, 1) @ y[:t0] for t0 in t0d]),
       "TM12": np.array([y[t0 - H:t0].mean() for t0 in t0d])}
yoy = 100 * (np.exp(logcpi[12:] - logcpi[:-12]) - 1)            # inflasi tahunan dari indeks, t = 13..n
tm_raw = np.array([12 * r_raw[t - 12:t].mean() for t in range(13, n + 1)])
tm_s2 = np.array([12 * r_s2[t - 12:t].mean() for t in range(13, n + 1)])
print(f"\n(v) DG-4 pada S2, tahunan = 12 x tren bulanan (poin persen). Cek: |12 x rata-rata trailing (mentah) - yoy| "
      f"maks {np.abs(tm_raw - yoy).max():.2f}, rata-rata {np.abs(tm_raw - yoy).mean():.2f}; |S2 - mentah| maks "
      f"{np.abs(tm_s2 - tm_raw).max():.2f}, rata-rata {np.abs(tm_s2 - tm_raw).mean():.2f}")
EPIS = {"2013-14": (37, 60), "2020": (121, 132), "2022": (145, 156), "all": (48, n - H)}
dg = []
for ep, (a, b) in EPIS.items():
    m = (t0d >= a) & (t0d <= b)
    rec = {"episode": ep, "months": int(m.sum()), "final_mean": 12 * final[m].mean(),
           "final_change": 12 * (final[m][-1] - final[m][0])}
    for e, v in est.items():
        gap = 12 * (v[m] - final[m])
        j = int(np.argmax(np.abs(gap)))
        rec.update({f"{e}_mean_gap": gap.mean(), f"{e}_max_abs_gap": gap[j],
                    f"{e}_max_abs_month": df.date.iloc[t0d[m][j] - 1].strftime("%Y-%m"),
                    f"{e}_rms_gap": np.sqrt(np.mean(gap**2))})
    dg.append(rec)
dg = pd.DataFrame(dg); dg.to_csv(OUT / "k6_dg4.csv", index=False)
print(dg.to_string(index=False, float_format=lambda v: f"{v:+.2f}" if isinstance(v, float) else str(v)))
for ep, (a, b) in EPIS.items():
    if ep == "all":
        continue
    m = (t0d >= a) & (t0d <= b)
    print(f"  {ep}: final tahunan {12 * final[m].min():.2f} sampai {12 * final[m].max():.2f}; yoy publikasi "
          f"{yoy[t0d[m] - 13].min():.2f} sampai {yoy[t0d[m] - 13].max():.2f}")

# ---------- (vi) Gambar 5: diagram alir ----------
B_LO, B_HI = round(BAND[1.0][0], 1), round(BAND[0.5][1], 1)
dot = f'''digraph G {{
  graph [rankdir=TB, nodesep=0.22, ranksep=0.22, fontname="DejaVu Sans", pad=0.1];
  node  [fontname="DejaVu Sans", fontsize=9, shape=box, style="rounded", margin="0.08,0.04", height=0.3];
  edge  [fontname="DejaVu Sans", fontsize=8, arrowsize=0.6];
  start [label="Monthly series; trend wanted at the latest month", style="rounded,filled", fillcolor="#eeeeee"];
  s1  [label="(i) Regress NW revisions on the local slope\\nand LL revisions on the local curvature"];
  d1  [shape=diamond, label="Ratios match\\nEq. (25a)?"];
  x1  [label="Inspect curvature,\\nbreaks, outliers", style="rounded,dashed"];
  s2  [label="(ii) NW lags by delta_h times the current slope\\n(h = 12: 4.2 months; 12-month mean: 5.5)"];
  s3  [label="(iii) ADF and KPSS with constant and linear trend, 5%"];
  d3  [shape=diamond, label="Stationary?"];
  d4  [shape=diamond, label="AR(1)\\nadequate?"];
  d5  [shape=diamond, label="rho_D1 <= 0.7?"];
  mcv [label="MCV with l*(rho_D1),\\nEq. (20); report s = (2l + 1)/(2nh)"];
  e26 [label="Own bandwidths by Eq. (26);\\nreference window 12 (18, 24)"];
  s4  [label="(iv) Stationary errors: SNR / SNR0"];
  b1  [label="below {B_LO}:\\nNW favoured"];
  b2  [label="{B_LO} to {B_HI}:\\nNW not dominated"];
  b3  [label="above {B_HI}: LL favoured unless\\nthe endpoint slope is weak"];
  start -> s1 -> d1;
  d1 -> x1 [label=" no"];
  d1 -> s2 [label=" yes"];
  s2 -> s3 -> d3;
  d3 -> d4 [label=" yes"];
  d3 -> e26 [label=" no: integrated\\n or conflicting"];
  d4 -> d5 [label=" yes"];
  d4 -> e26 [label=" no"];
  d5 -> mcv [label=" yes"];
  d5 -> e26 [label=" no"];
  mcv -> s4;
  e26 -> s4 [style=dashed, label=" if stationary"];
  s4 -> b1; s4 -> b2; s4 -> b3;
  {{rank=same; d1; x1}}
  {{rank=same; mcv; e26}}
  {{rank=same; b1; b2; b3}}
}}
'''
dot = (dot.replace("delta_h", "δₕ").replace("rho_D1", "ρ̂")
       .replace("SNR0", "SNR₀").replace("l*(", "ℓ*(").replace("(2l + 1)", "(2ℓ̂ + 1)")
       .replace("s = (", "ŝ = (").replace("/(2nh)", "/(2nĥ)")
       .replace("<= 0.7", "≤ 0.7").replace("r1 of", "r₁ of"))
(OUT / "fig5_flowchart.dot").write_text(dot, encoding="utf8")
if shutil.which("dot"):
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", str(OUT / "fig5_flowchart.dot"), "-o", str(OUT / "fig5_flowchart.png")],
                   check=True)
    print(f"\n(vi) [tersimpan] {OUT / 'fig5_flowchart.png'} dan sumbernya fig5_flowchart.dot")
else:
    print(f"\n(vi) program dot tidak ada; sumber Graphviz tersimpan di {OUT / 'fig5_flowchart.dot'}")
print(f"\nselesai dalam {time.time() - t_start:.0f} s; [tersimpan] output/k6_*.csv")
