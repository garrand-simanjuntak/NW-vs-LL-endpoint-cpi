"""
07_mse_plot.py
Visualisasi kurva MISE vs h dan dekomposisi bias^2/varians, NW vs Local Linear.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lpr_core import exact_bias, exact_variance, ar1_corr
from sim_functions import TARGETS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
K = "epanechnikov"
SIGMA = 0.30
H = np.linspace(0.03, 0.28, 40)
Sig = ar1_corr(n, 0.7)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, tgt in enumerate(TARGETS):
    for row, (Sg, lab) in enumerate([(None, "i.i.d."), (Sig, "AR(1) rho=0.7")]):
        ax = axes[row, j]
        mise = {0: [], 1: []}
        for h in H:
            for p in (0, 1):
                b = exact_bias(X, X, h, p, tgt, K)
                v = exact_variance(X, X, h, p, SIGMA**2, K, Sigma=Sg)
                mise[p].append(np.nanmean(b**2 + v))
        mise = {p: np.array(mise[p]) for p in (0, 1)}
        ax.plot(H, mise[0], color="#c0504d", lw=2, label="NW (p=0)")
        ax.plot(H, mise[1], color="#2e7d32", lw=2, label="LL (p=1)")
        for p, c in [(0, "#c0504d"), (1, "#2e7d32")]:
            i = np.argmin(mise[p])
            ax.plot(H[i], mise[p][i], "o", color=c, ms=8, mec="k", mew=.8)
            ax.annotate(f"h*={H[i]:.3f}", (H[i], mise[p][i]),
                        textcoords="offset points", xytext=(6, 8), fontsize=8, color=c)
        ax.set_yscale("log"); ax.grid(alpha=.3, which="both")
        ax.set_title(f"{tgt.name} - {lab}", fontsize=10, weight="bold")
        if row == 1:
            ax.set_xlabel("bandwidth h")
        if j == 0:
            ax.set_ylabel("MISE (skala log)")
        if row == 0 and j == 0:
            ax.legend(fontsize=9)

fig.suptitle("MISE = Bias^2(x) + Var(x), dirata-ratakan - NW vs Local Linear, desain tetap n=198",
             fontsize=12)
plt.tight_layout()
out_path = OUT_DIR / "sim_mise_curves.png"
plt.savefig(out_path, dpi=135)
print(f"[tersimpan] {out_path}")
