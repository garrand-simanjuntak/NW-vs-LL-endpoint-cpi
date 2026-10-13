"""
04_mc_validation_plot.py
(1) Validasi Monte Carlo dengan kriteria uji yang benar (memperhitungkan
    multiple testing di 198 titik evaluasi).
(2) Visualisasi kurva bias NW vs LL di seluruh domain.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from lpr_core import exact_bias, weight_matrix
from sim_functions import TARGETS, M1

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(20260711)

n = 198
X = np.arange(1, n + 1) / n
KERNEL = "epanechnikov"
SIGMA = 0.30
R = 2000
h = 0.10

alpha = 0.05
z_crit = norm.ppf(1 - alpha / (2 * n))
print(f"Ambang kritis (Bonferroni, alpha={alpha}, {n} titik): z = {z_crit:.3f}\n")

print("=" * 78)
print(f"VALIDASI MONTE CARLO (R={R}, h={h}, sigma={SIGMA})")
print("Uji: apakah bias empirik konsisten dengan bias eksak dalam galat MC?")
print("=" * 78)

for tgt in TARGETS:
    mX = tgt(X)
    L0 = weight_matrix(X, X, h, 0, KERNEL)
    L1 = weight_matrix(X, X, h, 1, KERNEL)

    acc0 = np.zeros(n); acc1 = np.zeros(n)
    for _ in range(R):
        Y = mX + rng.normal(0, SIGMA, n)
        acc0 += L0 @ Y
        acc1 += L1 @ Y

    for lbl, L, acc in [("NW", L0, acc0), ("LL", L1, acc1)]:
        bias_mc = acc / R - tgt(X)
        bias_ex = exact_bias(X, X, h, 0 if lbl == "NW" else 1, tgt, KERNEL)
        se = SIGMA * np.sqrt(np.sum(L**2, axis=1) / R)
        z = np.abs(bias_mc - bias_ex) / se
        n_viol = int(np.sum(z > z_crit))
        print(f"  {tgt.name:8s} {lbl}: max|z| = {z.max():5.2f} | pelanggaran = {n_viol}/{n}")

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

for j, tgt in enumerate(TARGETS):
    ax = axes[0, j]
    ax.plot(X, tgt(X), color="#1f4e79", lw=1.8)
    ax.set_title(f"{tgt.name}", fontsize=11, weight="bold")
    ax.set_ylabel("m(x)" if j == 0 else "")
    ax.grid(alpha=.3)

    ax = axes[1, j]
    for hh, ls in zip([0.05, 0.10, 0.15], ["-", "--", ":"]):
        b_nw = exact_bias(X, X, hh, 0, tgt, KERNEL)
        b_ll = exact_bias(X, X, hh, 1, tgt, KERNEL)
        ax.plot(X, b_nw, ls, color="#c0504d", lw=1.5,
                label=f"NW h={hh}" if j == 0 else None)
        ax.plot(X, b_ll, ls, color="#2e7d32", lw=1.5,
                label=f"LL h={hh}" if j == 0 else None)
    ax.axhline(0, color="k", lw=.6)
    ax.axvspan(0, 0.15, color="grey", alpha=.12)
    ax.axvspan(0.85, 1.0, color="grey", alpha=.12)
    ax.set_xlabel("x"); ax.set_ylabel("Bias(x)" if j == 0 else "")
    ax.grid(alpha=.3)
    if j == 0:
        ax.legend(fontsize=7, ncol=2)

fig.suptitle("Bias eksak NW (merah) vs Local Linear (hijau) - desain tetap, n=198\n"
             "Area abu-abu = wilayah batas", fontsize=12)
plt.tight_layout()
out_path = OUT_DIR / "sim_bias_curves.png"
plt.savefig(out_path, dpi=135)
print(f"\n[tersimpan] {out_path}")

print()
print("=" * 78)
print("LAJU KONVERGENSI BIAS DI TITIK BATAS x=1")
print("Teori: NW = O(h)  [orde 1]   |   LL = O(h^2)  [orde 2]")
print("=" * 78)

hs = np.array([0.20, 0.15, 0.10, 0.075, 0.05, 0.035])
rows = []
for tgt in TARGETS:
    b_nw = np.array([exact_bias(np.array([1.0]), X, hh, 0, tgt, KERNEL)[0] for hh in hs])
    b_ll = np.array([exact_bias(np.array([1.0]), X, hh, 1, tgt, KERNEL)[0] for hh in hs])
    ord_nw = np.polyfit(np.log(hs), np.log(np.abs(b_nw)), 1)[0]
    ord_ll = np.polyfit(np.log(hs), np.log(np.abs(b_ll)), 1)[0]
    rows.append({"target": tgt.name,
                 "orde_NW_empirik": ord_nw,
                 "orde_LL_empirik": ord_ll,
                 "m'(1)": float(tgt.dm(np.array([1.0]))[0])})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
