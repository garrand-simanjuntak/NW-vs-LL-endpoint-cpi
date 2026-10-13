"""
fig2_mise_curves.py
Figure 2 -- MISE against bandwidth, m(x) = exp(2x), fixed design n = 198:
(a) i.i.d. errors; (b) AR(1) errors with rho = 0.7.

REVISION (per supervisor):
  * panel labels (a)/(b) moved BELOW the panels;
  * all in-figure text in English (was: galat i.i.d., galat AR(1), skala log);
  * in-figure suptitle removed.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_style import panel_labels_below
from lpr_core import exact_bias, exact_variance, ar1_corr
from sim_functions import M2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
K = "epanechnikov"
SIGMA = 0.30
RHO = 0.7
H = np.linspace(0.03, 0.28, 40)
SIG = ar1_corr(n, RHO)

REGIMES = [
    (None, "(a) i.i.d. errors"),
    (SIG,  r"(b) AR(1) errors, $\rho = 0.7$"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, (Sg, _) in zip(axes, REGIMES):
    mise = {0: [], 1: []}
    for h in H:
        for p in (0, 1):
            b = exact_bias(X, X, h, p, M2, K)
            v = exact_variance(X, X, h, p, SIGMA**2, K, Sigma=Sg)
            mise[p].append(np.nanmean(b**2 + v))
    mise = {p: np.array(mise[p]) for p in (0, 1)}

    ax.plot(H, mise[0], color="#c0504d", lw=2, label="NW (p = 0)")
    ax.plot(H, mise[1], color="#2e7d32", lw=2, label="LL (p = 1)")
    for p, c in [(0, "#c0504d"), (1, "#2e7d32")]:
        i = int(np.argmin(mise[p]))
        ax.plot(H[i], mise[p][i], "o", color=c, ms=8, mec="k", mew=.8)
        ax.annotate(rf"$h^* = {H[i]:.3f}$", (H[i], mise[p][i]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8.5, color=c)
    ax.set_yscale("log")
    ax.grid(alpha=.3, which="both")
    ax.set_xlabel("bandwidth $h$")

axes[0].set_ylabel("MISE (log scale)")
axes[0].legend(loc="best")

plt.tight_layout(rect=[0, 0.135, 1, 1])
panel_labels_below(fig, axes, [lab for _, lab in REGIMES], dy=0.105)

p = OUT_DIR / "fig2_mise_curves.png"
plt.savefig(p, dpi=300, bbox_inches="tight")
print(f"[saved] {p}")
