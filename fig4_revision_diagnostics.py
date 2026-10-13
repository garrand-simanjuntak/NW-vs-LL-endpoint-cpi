"""
fig4_revision_diagnostics.py
Figure 4 -- Revision diagnostics on the reconstructed Indonesian CPI
(BPS, 2010-2026, n = 198), h = 12 months. Menggantikan fig3_revision_analysis.py.

ALASAN REVISI (Reviewer 1, komentar Test (A) dan Gambar 3 lama)
Gambar lama memakai garis asimtotik 0.375 h m'(t0), padahal Bagian 2.9 dan
Tabel 5 menetapkan prediksi diskret eksak delta_h sebagai patokan yang benar.
Gambar lama juga memplot revisi LL terhadap kemiringan, padahal revisi LL
digerakkan kurvatur.

PANEL
 (a) log(CPI): revisi NW vs m'(t0), garis delta_h (utuh) dan 0.375h (titik-titik)
 (b) log(CPI): revisi LL vs m''(t0), garis gamma_h
 (c), (d): sama untuk inflasi bulanan
m'(t0) dari fit LL sampel penuh, m''(t0) dari fit kuadratik lokal sampel penuh,
keduanya pada h yang sama.
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
H = 12
T0_MIN = 48
C_NW, C_LL = "#c0504d", "#2e7d32"


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def rows(t0, Tv, h, p):
    w = epan((Tv - t0) / h)
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


k = np.arange(H); Kk = 1 - (k / H) ** 2
DELTA = float((Kk * k).sum() / Kk.sum())                       # Eq. (25)
t_mid = n // 2; d = T - t_mid
GAMMA = 0.5 * ((rows(t_mid, T, H, 1)[0] * d**2).sum()
               - (rows(t_mid, T[:t_mid], H, 1)[0] * d[:t_mid] ** 2).sum())

RESPONSES = [("log_IHK", "log(CPI)"), ("infl_pct_mtm", "Month-to-month inflation")]
data = []
for col, lab in RESPONSES:
    Y = df[col].values
    fit = np.array([rows(t, T, 12, 1)[0] @ Y for t in T])
    snr = fit.std() / (Y - fit).std()
    slope, curv, rn, rl = [], [], [], []
    for t0 in range(T0_MIN, n - H + 1):
        fin = rows(t0, T, H, 1)[0] @ Y
        rn.append(fin - rows(t0, T[:t0], H, 0)[0] @ Y[:t0])
        rl.append(fin - rows(t0, T[:t0], H, 1)[0] @ Y[:t0])
        slope.append(rows(t0, T, H, 1)[1] @ Y)
        curv.append(2 * rows(t0, T, H, 2)[2] @ Y)
    data.append((lab, snr, *map(np.array, (slope, curv, rn, rl))))

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
labels = []
for i, (lab, snr, slope, curv, rn, rl) in enumerate(data):
    ax = axes[i, 0]
    ax.scatter(slope, rn, s=14, color=C_NW, alpha=.7, label="NW revision")
    xs = np.linspace(slope.min(), slope.max(), 50)
    ax.plot(xs, DELTA * xs, "k-", lw=1.6,
            label=rf"exact discrete: $\delta_h\,\hat m'(t_0)$, $\delta_{{{H}}}={DELTA:.3f}$")
    ax.plot(xs, 0.375 * H * xs, color="grey", ls=":", lw=1.6,
            label=rf"asymptotic: $0.375\,h\,\hat m'(t_0)$")
    ax.axhline(0, color="grey", lw=.7)
    ax.set_xlabel(r"local slope $\hat m'(t_0)$")
    ax.set_ylabel("Revision")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    labels.append(f"({'ac'[i]}) {lab}, NW revisions (SNR = {snr:.2f})")

    ax = axes[i, 1]
    ax.scatter(curv, rl, s=14, color=C_LL, alpha=.7, label="LL revision")
    xs = np.linspace(curv.min(), curv.max(), 50)
    ax.plot(xs, GAMMA * xs, "k-", lw=1.6,
            label=rf"exact discrete: $\gamma_h\,\hat m''(t_0)$, $\gamma_{{{H}}}={GAMMA:.2f}$")
    ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7)
    ax.set_xlabel(r"local curvature $\hat m''(t_0)$")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    labels.append(f"({'bd'[i]}) {lab}, LL revisions")

plt.tight_layout(rect=[0, 0.03, 1, 1], h_pad=4.5)
panel_labels_below(fig, axes.ravel(), labels, dy=0.065)
p = OUT_DIR / "fig4_revision_diagnostics.png"
plt.savefig(p, dpi=300, bbox_inches="tight")
print(f"delta_{H} = {DELTA:.4f}, gamma_{H} = {GAMMA:.3f}")
print(f"[saved] {p}")
