"""
09_equivalent_kernel.py
Mekanisme struktural: MENGAPA Local Linear mengoreksi bias batas.

Menampilkan bobot ekuivalen w_t(x) dan kondisi momen diskretnya.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lpr_core import lp_weights
from sim_functions import M2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198; X = np.arange(1, n+1)/n; h = 0.10; K = "epanechnikov"
pts = [(0.50, "Interior (x=0.50)"), (0.95, "Batas (x=0.95)"), (1.00, "Batas ekstrem (x=1.00)")]

print("KONDISI MOMEN DISKRET  —  sum w=1 (reproduksi konstanta), sum w(X-x)=0 (reproduksi linear)")
print(f"{'titik':>24} {'est':>4} {'sum w':>10} {'sum w(X-x)':>13}")
for x0, lbl in pts:
    for p, e in [(0, "NW"), (1, "LL")]:
        w = lp_weights(x0, X, h, p, K)
        print(f"{lbl:>24} {e:>4} {w.sum():10.6f} {(w*(X-x0)).sum():13.3e}")

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for j, (x0, lbl) in enumerate(pts):
    w0 = lp_weights(x0, X, h, 0, K)
    w1 = lp_weights(x0, X, h, 1, K)
    sel = np.abs(X - x0) <= h * 1.15

    ax = axes[0, j]
    ax.axhline(0, color="k", lw=.7)
    ax.axvline(x0, color="grey", ls=":", lw=1)
    ax.plot(X[sel], w0[sel], "o-", color="#c0504d", ms=3, lw=1.3, label="NW (p=0)")
    ax.plot(X[sel], w1[sel], "s-", color="#2e7d32", ms=3, lw=1.3, label="LL (p=1)")
    ax.fill_between(X[sel], 0, w1[sel], where=(w1[sel] < 0), color="#2e7d32", alpha=.25)
    ax.set_title(lbl, fontsize=11, weight="bold")
    ax.set_ylabel("bobot $w_t(x)$" if j == 0 else "")
    ax.grid(alpha=.3)
    if j == 0: ax.legend(fontsize=8)

    ax = axes[1, j]
    m = M2(X)
    ax.plot(X[sel], m[sel], color="#1f4e79", lw=1.5, label="m(x)")
    ax.plot(x0, M2(np.array([x0]))[0], "k*", ms=13, label="target m(x$_0$)")
    ax.plot(x0, w0 @ m, "o", color="#c0504d", ms=8, label="E[NW]")
    ax.plot(x0, w1 @ m, "s", color="#2e7d32", ms=7, label="E[LL]")
    ax.axvline(x0, color="grey", ls=":", lw=1)
    ax.set_xlabel("x"); ax.set_ylabel("m(x)" if j == 0 else "")
    ax.grid(alpha=.3)
    if j == 0: ax.legend(fontsize=8)

fig.suptitle("Bobot ekuivalen dan konsekuensinya (M2_exp, h=0.10)\n"
             "Area hijau = bobot LL NEGATIF — inilah mekanisme koreksinya", fontsize=12)
plt.tight_layout()
p = OUT_DIR / "equivalent_kernel.png"
plt.savefig(p, dpi=135)
print(f"\n[tersimpan] {p}")
