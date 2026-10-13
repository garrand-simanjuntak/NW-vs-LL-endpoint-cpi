"""
fig1_equivalent_weights.py
Figure 1 -- Equivalent weights of the NW and LL estimators at three
evaluation points, m(x) = exp(2x), h = 0.10.

REVISION (per supervisor):
  * panel labels (a)/(b)/(c) moved BELOW the panels;
  * all in-figure text in English (was: bobot, Interior, Batas, Batas ekstrem);
  * in-figure suptitle removed -- the caption belongs in the manuscript.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_style import panel_labels_below
from lpr_core import lp_weights
from sim_functions import M2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
h = 0.10
K = "epanechnikov"

# evaluation points and their panel labels (English)
PTS = [
    (0.50, "(a) Interior (x = 0.50)"),
    (0.95, "(b) Boundary (x = 0.95)"),
    (1.00, "(c) Extreme boundary (x = 1.00)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

for j, (x0, _) in enumerate(PTS):
    w0 = lp_weights(x0, X, h, 0, K)
    w1 = lp_weights(x0, X, h, 1, K)
    sel = np.abs(X - x0) <= h * 1.15

    # --- top row: equivalent weights -------------------------------------
    ax = axes[0, j]
    ax.axhline(0, color="k", lw=.7)
    ax.axvline(x0, color="grey", ls=":", lw=1)
    ax.plot(X[sel], w0[sel], "o-", color="#c0504d", ms=3, lw=1.3, label="NW (p = 0)")
    ax.plot(X[sel], w1[sel], "s-", color="#2e7d32", ms=3, lw=1.3, label="LL (p = 1)")
    ax.fill_between(X[sel], 0, w1[sel], where=(w1[sel] < 0), color="#2e7d32", alpha=.25)
    ax.set_ylabel("weight $w_t(x)$" if j == 0 else "")
    ax.grid(alpha=.3)
    if j == 0:
        ax.legend(loc="best")

    # --- bottom row: consequence for E[m_hat] ----------------------------
    ax = axes[1, j]
    m = M2(X)
    ax.plot(X[sel], m[sel], color="#1f4e79", lw=1.5, label="$m(x)$")
    ax.plot(x0, M2(np.array([x0]))[0], "k*", ms=13, label="target $m(x_0)$")
    ax.plot(x0, w0 @ m, "o", color="#c0504d", ms=8, label=r"$E[\hat{m}_{NW}]$")
    ax.plot(x0, w1 @ m, "s", color="#2e7d32", ms=7, label=r"$E[\hat{m}_{LL}]$")
    ax.axvline(x0, color="grey", ls=":", lw=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$m(x)$" if j == 0 else "")
    ax.grid(alpha=.3)
    if j == 0:
        ax.legend(loc="best")

plt.tight_layout(rect=[0, 0.115, 1, 1])
panel_labels_below(fig, axes[1, :], [lab for _, lab in PTS], dy=0.092)

p = OUT_DIR / "fig1_equivalent_weights.png"
plt.savefig(p, dpi=300, bbox_inches="tight")
print(f"[saved] {p}")
