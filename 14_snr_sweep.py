"""
14_snr_sweep.py
Sweep SNR untuk menemukan titik crossover NW vs LL di titik ujung, dengan
setiap estimator pada bandwidth optimalnya sendiri (menjawab Reviewer 1,
komentar Data/SNR, Test (B), dan Kesimpulan).

TEORI (asimtotik, titik ujung kanan c = 0, kernel Epanechnikov)
    MSE_NW(h) = (B0 h m')^2     + sigma^2 V0 / (n h),   B0 = 3/8,    V0 = 1.2
    MSE_LL(h) = (B1 h^2 m'')^2  + sigma^2 V1 / (n h),   B1 = 0.0579, V1 = 4.498
Masing-masing diminimumkan terhadap h:
    MSE_NW* = 3 a^{1/3} (b/2)^{2/3},   MSE_LL* = 5 c^{1/5} (b'/4)^{4/5}
Rasio MSE_NW*/MSE_LL* hanya bergantung pada satu indeks tak berdimensi
    Lambda = sqrt(n) |m'(1)|^{5/2} / ( |m''(1)|^{3/2} sigma ),
yaitu rasio = C Lambda^{4/15}, sehingga crossover terjadi di Lambda* ~ 5.60.
Dalam satuan waktu (x = t/n) faktor n saling menghapus:
    Lambda = |m'_t|^{5/2} / ( |m''_t|^{3/2} sigma ).
Untuk galat AR(1), sigma diganti SD jangka panjang sigma sqrt((1+rho)/(1-rho)).

SIMULASI (eksak, tanpa Monte Carlo)
Pada desain tetap, bias = w'm - m(1) dan Var = sigma^2 w' Sigma w dihitung
eksak dari bobot ekuivalen di x = 1. Untuk tiap fungsi target, sigma digeser,
h* tiap estimator dicari pada grid halus, lalu crossover sigma* (rasio = 1)
diinterpolasi. SNR global didefinisikan sd_t(m(x_t)) / sigma.

Revisi putaran 2 (Reviewer 2, butir HA-1 dan HA-4):
  - Sweep diulang dengan batas atas bandwidth h_max in {1.0, 0.5, 0.3}
    (kolom h_max di kedua CSV). h_max = 1.0 identik dengan versi sebelumnya.
  - Gambar (Figure 3 naskah): warna per keluarga target, gaya garis per anggota,
    dua target s = 1 (tanpa crossover) diberi penanda dan label, garis SNR0 dan
    deret empiris berlabel di legenda.

Keluaran: output/snr_sweep_curves.csv, output/snr_crossover.csv,
          output/snr_real_lambda.csv, output/fig4_snr_crossover.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lpr_core import lp_weights, ar1_corr, weight_matrix

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

N_LIST = [198, 400]          # n = 100 dibuang: jendela minimal memicu crossover semu di sigma sangat kecil
RHOS = [0.0, 0.3, 0.6]
SIGMAS = np.logspace(-3, 2.5, 160)
H_MAX_LIST = [1.0, 0.5, 0.3]   # 1.0 = seluruh rentang (versi naskah); 0.5 dan 0.3 = sensitivitas jendela lokal


# ---------- konstanta asimtotik dan Lambda* tertutup ----------
def _K(u):
    return 0.75 * (1 - u * u)


nu = [quad(lambda u, j=j: u**j * _K(u), -1, 0)[0] for j in range(4)]
B0 = abs(nu[1] / nu[0])
V0 = quad(lambda u: _K(u) ** 2, -1, 0)[0] / nu[0] ** 2
DEN = nu[0] * nu[2] - nu[1] ** 2
B1 = abs(0.5 * (nu[2] ** 2 - nu[1] * nu[3]) / DEN)
V1 = quad(lambda u: ((nu[2] - nu[1] * u) * _K(u) / DEN) ** 2, -1, 0)[0]


def asym_ratio(lam):
    """Rasio MSE_NW*/MSE_LL* asimtotik sebagai fungsi Lambda."""
    a = B0**2 * lam ** 0.8              # m'^2 dengan m'^{5/2} = Lambda, m''=sigma=n=1
    nw = 3 * a ** (1 / 3) * (V0 / 2) ** (2 / 3)
    ll = 5 * B1 ** 0.4 * (V1 / 4) ** 0.8
    return nw / ll


LAMBDA_STAR = brentq(lambda L: asym_ratio(L) - 1, 1e-6, 1e6)
C_RATIO = asym_ratio(1.0)

# ---------- fungsi target (m, m'(1), m''(1)) ----------
TARGETS = []
for a in [1.0, 2.0, 3.0]:
    TARGETS.append((f"exp({a:g}x)", lambda x, a=a: np.exp(a * x),
                    a * np.exp(a), a * a * np.exp(a)))
for phi in [0.3, 0.7, 1.1]:
    TARGETS.append((f"sin(2pi x+{phi:g})",
                    lambda x, p=phi: np.sin(2 * np.pi * x + p),
                    2 * np.pi * np.cos(phi), -4 * np.pi**2 * np.sin(phi)))
for s, q in [(1.0, 4.0), (1.0, 12.0), (3.0, 4.0)]:
    TARGETS.append((f"{s:g}x+{q:g}/2(x-1)^2",
                    lambda x, s=s, q=q: s * x + 0.5 * q * (x - 1) ** 2, s, q))


def paper_snr(mX, sig, L, Sig):
    """Pendekatan deterministik E[sd(fit)]/E[sd(resid)] untuk SNR naskah
    (fit LL h = 12 bulan pada seluruh deret, Bagian 2.8)."""
    n = len(mX); I = np.eye(n); C = I - np.ones((n, n)) / n
    S = I if Sig is None else Sig
    def v(A):
        return (np.var(A @ mX) + sig**2 * np.trace(C @ A @ S @ A.T @ C) / n)
    return np.sqrt(v(L) / v(I - L))


def endpoint_weights(n):
    X = np.arange(1, n + 1) / n
    hs = np.unique(np.concatenate([np.arange(3, n + 1) / n]))
    W = {}
    for p in (0, 1):
        rows, hv = [], []
        for h in hs:
            w = lp_weights(1.0, X, h, p)
            if np.all(np.isfinite(w)) and (w != 0).sum() >= p + 2:
                rows.append(w); hv.append(h)
        W[p] = (np.array(hv), np.vstack(rows))
    return X, W


curve_rows, cross_rows = [], []
for n in N_LIST:
    X, W = endpoint_weights(n)
    for rho in RHOS:
        Sig = ar1_corr(n, rho) if rho > 0 else None
        quad_form = {p: (np.einsum("ij,jk,ik->i", W[p][1], Sig, W[p][1]) if Sig is not None
                         else np.sum(W[p][1] ** 2, axis=1)) for p in (0, 1)}
        lr = np.sqrt((1 + rho) / (1 - rho))
        L12 = weight_matrix(X, X, 12 / n, 1)
        for name, m, d1, d2 in TARGETS:
            mX, m1 = m(X), m(1.0)
            bias2 = {p: (W[p][1] @ mX - m1) ** 2 for p in (0, 1)}
            sd_m = mX.std()
            for h_max in H_MAX_LIST:
                keep = {p: W[p][0] <= h_max + 1e-12 for p in (0, 1)}
                ratios = []
                for sig in SIGMAS:
                    crit = {p: (bias2[p] + sig**2 * quad_form[p])[keep[p]] for p in (0, 1)}
                    mse = {p: np.min(crit[p]) for p in (0, 1)}
                    hst = {p: W[p][0][keep[p]][np.argmin(crit[p])] for p in (0, 1)}
                    r = mse[0] / mse[1]
                    ratios.append(r)
                    lam = np.sqrt(n) * abs(d1) ** 2.5 / (abs(d2) ** 1.5 * sig * lr)
                    curve_rows.append({"h_max": h_max, "n": n, "rho": rho, "target": name,
                                       "sigma": sig, "SNR_global": sd_m / sig, "Lambda": lam,
                                       "ratio_NW_LL": r, "h_NW": hst[0], "h_LL": hst[1]})
                ratios = np.array(ratios)
                s = np.where(np.diff(np.sign(np.log(ratios))) != 0)[0]
                if len(s) != 1:
                    cross_rows.append({"h_max": h_max, "n": n, "rho": rho, "target": name,
                                       "n_cross": len(s),
                                       "NW_always_better": bool(np.all(ratios < 1))})
                    continue
                i = s[0]
                f = np.log(ratios[i]) / (np.log(ratios[i]) - np.log(ratios[i + 1]))
                sig_star = np.exp(np.log(SIGMAS[i]) + f * (np.log(SIGMAS[i + 1]) - np.log(SIGMAS[i])))
                lam_star = np.sqrt(n) * abs(d1) ** 2.5 / (abs(d2) ** 1.5 * sig_star * lr)
                hs_nw = np.interp(sig_star, SIGMAS, [c["h_NW"] for c in curve_rows[-len(SIGMAS):]])
                hs_ll = np.interp(sig_star, SIGMAS, [c["h_LL"] for c in curve_rows[-len(SIGMAS):]])
                cross_rows.append({"h_max": h_max, "n": n, "rho": rho, "target": name,
                                   "n_cross": 1, "NW_always_better": False,
                                   "SNR_paper_star": paper_snr(mX, sig_star, L12, Sig),
                                   "sigma_star": sig_star, "SNR_global_star": sd_m / sig_star,
                                   "Lambda_star": lam_star, "h_NW_at_star": hs_nw,
                                   "h_LL_at_star": hs_ll})

curves = pd.DataFrame(curve_rows)
cross = pd.DataFrame(cross_rows)
curves.to_csv(OUT_DIR / "snr_sweep_curves.csv", index=False)
cross.to_csv(OUT_DIR / "snr_crossover.csv", index=False)

pd.set_option("display.width", 200)
print(f"Konstanta: B0={B0:.4f} V0={V0:.4f} B1={B1:.4f} V1={V1:.4f}")
print(f"Crossover asimtotik tertutup: Lambda* = {LAMBDA_STAR:.3f}  (rasio = {C_RATIO:.4f} Lambda^(4/15))\n")
print("CROSSOVER EKSAK (desain tetap, h optimal masing-masing)")
print(cross[cross.n_cross == 1].drop(columns=["n_cross", "NW_always_better"])
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nRingkasan crossover per (n, rho): SNRp = skala SNR naskah, spread = max/min")
ok = cross[cross.n_cross == 1]
print("\nTanpa crossover tunggal (NW_always_better = True berarti NW unggul di semua sigma):")
print(cross[cross.n_cross != 1][["h_max", "n", "rho", "target", "n_cross", "NW_always_better"]]
      .to_string(index=False))
summ = ok.groupby(["h_max", "n", "rho"]).agg(
    n_target=("target", "count"),
    SNRp_min=("SNR_paper_star", "min"), SNRp_max=("SNR_paper_star", "max"),
    SNR_min=("SNR_global_star", "min"), SNR_max=("SNR_global_star", "max"),
    Lam_min=("Lambda_star", "min"), Lam_max=("Lambda_star", "max"),
    hLL_min=("h_LL_at_star", "min"), hLL_max=("h_LL_at_star", "max"),
    hNW_min=("h_NW_at_star", "min"), hNW_max=("h_NW_at_star", "max"))
summ["SNRp_spread"] = summ.SNRp_max / summ.SNRp_min
summ["SNR_spread"] = summ.SNR_max / summ.SNR_min
summ["Lam_spread"] = summ.Lam_max / summ.Lam_min
print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

# ---------- pemetaan data riil ----------
df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv")
T = np.arange(1, len(df) + 1, dtype=float)
H_PILOT = 18


def lq_rows(t0, h):
    w = np.where(np.abs((T - t0) / h) <= 1, 0.75 * (1 - ((T - t0) / h) ** 2), 0.0)
    Z = np.vander(T - t0, N=3, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


real_rows, snr_emp = [], {}
for name in ["log_IHK", "infl_pct_mtm"]:
    Y = df[name].values
    fit12 = np.array([lp_weights(t / len(T), T / len(T), 12 / len(T), 1) @ Y for t in T])
    sig = (Y - fit12).std()
    snr_emp[name] = fit12.std() / sig          # SNR empiris (definisi naskah)
    for t0 in range(48, len(T) - H_PILOT + 1):
        R = lq_rows(t0, H_PILOT)
        d1, d2 = R[1] @ Y, 2 * R[2] @ Y
        lam = abs(d1) ** 2.5 / (abs(d2) ** 1.5 * sig)
        real_rows.append({"response": name, "t0": t0, "m1": d1, "m2": d2,
                          "sigma": sig, "Lambda": lam})
real = pd.DataFrame(real_rows)
real.to_csv(OUT_DIR / "snr_real_lambda.csv", index=False)
print("\nSNR empiris (LL h = 12 bulan): " + ", ".join(f"{k} = {v:.3f}" for k, v in snr_emp.items()))
print("\nPEMETAAN DATA RIIL (pilot kuadratik lokal h = 18 bulan; sigma dari residual LL h = 12)")
for name, g in real.groupby("response", sort=False):
    q = g.Lambda.quantile([0.25, 0.5, 0.75]).values
    print(f"  {name:13s} Lambda kuartil = {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g}   "
          f"porsi t0 dengan Lambda > Lambda* = {(g.Lambda > LAMBDA_STAR).mean():.2f}")

# ---------- gambar ----------
sub = curves[(curves.h_max == 1.0) & (curves.n == 198) & (curves.rho == 0.0)].copy()
X198 = np.arange(1, 199) / 198; L198 = weight_matrix(X198, X198, 12 / 198, 1)
TF = {t[0]: t[1] for t in TARGETS}
sub["SNR_paper"] = [paper_snr(TF[r.target](X198), r.sigma, L198, None) for r in sub.itertuples()]
SNR0_IID = paper_snr(np.zeros(198), 1.0, L198, None)          # = Persamaan (22), galat i.i.d.

# Warna per keluarga target (palet kategorikal tervalidasi CVD), gaya garis per anggota.
# Dua target s = 1 tidak punya crossover: diberi penanda berongga dan keterangan di legenda.
C_EXP, C_SIN, C_QUAD = "#2a78d6", "#eb6834", "#1baf7a"
DASH, DOT, DASHDOT = (0, (4, 2)), (0, (1, 1.6)), (0, (6, 1.6, 1.2, 1.6))
STYLE = {  # target: (label, warna, gaya garis, penanda)
    "exp(1x)":        (r"$e^{x}$",                     C_EXP, "-", None),
    "exp(2x)":        (r"$e^{2x}$",                    C_EXP, DASH, None),
    "exp(3x)":        (r"$e^{3x}$",                    C_EXP, DASHDOT, None),
    "sin(2pi x+0.3)": (r"$\sin(2\pi x+0.3)$",          C_SIN, "-", None),
    "sin(2pi x+0.7)": (r"$\sin(2\pi x+0.7)$",          C_SIN, DASH, None),
    "sin(2pi x+1.1)": (r"$\sin(2\pi x+1.1)$",          C_SIN, DASHDOT, None),
    "3x+4/2(x-1)^2":  (r"$3x+2(x-1)^2$",               C_QUAD, "-", None),
    "1x+4/2(x-1)^2":  (r"$x+2(x-1)^2$ (no crossover)", C_QUAD, DASH, "o"),
    "1x+12/2(x-1)^2": (r"$x+6(x-1)^2$ (no crossover)", C_QUAD, DOT, "^"),
}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), sharey=True)
fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.37, wspace=0.08)
target_handles = []
for name in STYLE:
    g = sub[sub.target == name]
    lab, col, ls, mk = STYLE[name]
    kw = dict(color=col, ls=ls, lw=1.2)
    if mk is not None:
        kw.update(marker=mk, markevery=14, ms=4.2, mfc="white", mew=0.9)
    line, = axes[0].plot(g.SNR_paper, g.ratio_NW_LL, label=lab, **kw)
    axes[1].plot(g.Lambda, g.ratio_NW_LL, **kw)
    target_handles.append(line)
lg = np.logspace(-3, 6, 200)
h_asym, = axes[1].plot(lg, asym_ratio(lg), color=INK, ls=(0, (5, 2)), lw=1.6,
                       label=rf"asymptotic $C\Lambda^{{4/15}}$ ($C={C_RATIO:.3f}$)")
h_snr0 = axes[0].axvline(SNR0_IID, color=MUTED, lw=1.0, ls=(0, (4, 2)),
                         label=rf"noise floor SNR$_0={SNR0_IID:.3f}$")
h_infl = axes[0].axvline(snr_emp["infl_pct_mtm"], color=INK, lw=0.9, ls=(0, (1, 1.4)),
                         label=f"inflation, SNR = {snr_emp['infl_pct_mtm']:.3f}")
h_lcpi = axes[0].axvline(snr_emp["log_IHK"], color=INK, lw=0.9, ls=(0, (6, 1.5, 1, 1.5)),
                         label=f"log CPI, SNR = {snr_emp['log_IHK']:.1f}")
h_lstar = axes[1].axvline(LAMBDA_STAR, color=MUTED, lw=1.0, ls=(0, (1, 1.4)),
                          label=rf"$\Lambda^{{*}}={LAMBDA_STAR:.2f}$")
for ax, xl in zip(axes, ["SNR", r"endpoint index $\Lambda$"]):
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(color="#d9d8d4", lw=0.5, which="major")
    ax.set_xlabel(xl)
axes[0].set_ylabel(r"MSE ratio NW/LL at own optimal $h$, $x=1$")
axes[0].set_title("(a)", fontsize=8.5, loc="left"); axes[1].set_title("(b)", fontsize=8.5, loc="left")
fig.legend(handles=target_handles, loc="lower center", bbox_to_anchor=(0.54, 0.085), ncol=3,
           frameon=False, columnspacing=1.4, handlelength=3.0, title="target function",
           title_fontsize=7)
fig.legend(handles=[h_snr0, h_infl, h_lcpi, h_asym, h_lstar], loc="lower center",
           bbox_to_anchor=(0.54, -0.01), ncol=3, frameon=False, columnspacing=1.4,
           handlelength=3.0)
p = OUT_DIR / "fig4_snr_crossover.png"
plt.savefig(p, dpi=400)
print(f"\n[tersimpan] {p}")
