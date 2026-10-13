"""
fig3_revision_analysis.py
Figure 3 -- Revision analysis on the reconstructed Indonesian CPI
(BPS, 2010-2026, n = 198), h = 12 months:
(a) log(CPI), SNR = 33.63; (b) month-to-month inflation, SNR = 0.29.

REVISION (per supervisor):
  * panel labels (a)/(b) moved BELOW the panels;
  * all in-figure text in English (was: Revisi, kemiringan lokal, teori);
  * in-figure suptitle removed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_style import panel_labels_below

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
T = np.arange(1, len(df) + 1, dtype=float)
n = len(df)
H_PLOT = 12          # bandwidth in months used for the figure
T0_MIN = 48


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def wvec(t0, Tv, h, p):
    w = epan((Tv - t0) / h)
    if (w > 0).sum() < p + 2:
        return None
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    try:
        return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]
    except np.linalg.LinAlgError:
        return None


def slope_at(t0, Tv, Y, h):
    w = epan((Tv - t0) / h)
    Z = np.vander(Tv - t0, N=2, increasing=True)
    ZtW = Z.T * w
    return np.linalg.solve(ZtW @ Z, ZtW @ Y)[1]


RESPONSES = [
    ("log_IHK", df.log_IHK.values, "(a) log(CPI)"),
    ("infl_pct_mtm", df.infl_pct_mtm.values, "(b) Month-to-month inflation"),
]

panels = []
for name, Y, base_lab in RESPONSES:
    fit = np.array([wvec(t, T, 12, 1) @ Y for t in T])
    sig = (Y - fit).std()
    snr = fit.std() / sig

    mps, rn, rl = [], [], []
    for t0 in range(T0_MIN, n - H_PLOT + 1):
        w0 = wvec(t0, T[:t0], H_PLOT, 0)
        w1 = wvec(t0, T[:t0], H_PLOT, 1)
        if w0 is None or w1 is None:
            continue
        fin = wvec(t0, T, H_PLOT, 1) @ Y
        mps.append(slope_at(t0, T, Y, H_PLOT))
        rn.append(fin - w0 @ Y[:t0])
        rl.append(fin - w1 @ Y[:t0])
    panels.append((np.array(mps), np.array(rn), np.array(rl),
                   f"{base_lab}  (SNR = {snr:.2f})"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (mps, rn, rl, _) in zip(axes, panels):
    ax.scatter(mps, rn, s=14, color="#c0504d", alpha=.7, label="NW revision")
    ax.scatter(mps, rl, s=14, color="#2e7d32", alpha=.7, label="LL revision")
    xs = np.linspace(mps.min(), mps.max(), 50)
    ax.plot(xs, 0.375 * H_PLOT * xs, "k--", lw=1.6,
            label=r"theory: $0.375\,h\,\hat{m}'(t_0)$")
    ax.axhline(0, color="grey", lw=.7)
    ax.set_xlabel(r"$\hat{m}'(t_0)$  --  local slope")
    ax.set_ylabel("Revision")
    ax.legend(loc="best")
    ax.grid(alpha=.3)

plt.tight_layout(rect=[0, 0.135, 1, 1])
panel_labels_below(fig, axes, [lab for *_, lab in panels], dy=0.105)

p = OUT_DIR / "fig3_revision_analysis.png"
plt.savefig(p, dpi=300, bbox_inches="tight")
print(f"[saved] {p}")
