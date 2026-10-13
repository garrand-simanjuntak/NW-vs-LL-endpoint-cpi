"""
10_revision_analysis.py
APLIKASI DATA RIIL — revision analysis pada IHK BPS (2010-2026), n=198.

GAGASAN
-------
Pada waktu t0, estimasi "real-time" hanya boleh memakai data 1..t0, sehingga
t0 adalah TITIK BATAS. Setelah data baru tiba, t0 menjadi INTERIOR dan dapat
diestimasi ulang. Selisihnya:

    Revisi(t0) = m_hat_final(t0) - m_hat_realtime(t0)

Karena pada desain equidistant NW == LL PERSIS di interior (terbukti di
08_boundary_theory.py), estimasi FINAL kedua estimator identik (diverifikasi
sampai ~3e-16). Jadi proxy kebenaran SAMA untuk keduanya -> perbandingan adil,
dan seluruh perbedaan revisi berasal dari estimasi real-time saja.

DUA UJI
-------
(A) UJI TEORI (mengisolasi BIAS):
    Bias_NW(batas) ~ -0.375*h*m'(t0)  =>  Revisi_NW ~ +0.375*h*m'(t0)
    Regresi Revisi_NW thd m'(t0) HARUS bermiring ~0.375h. Uji ini dapat gagal.

(B) UJI AKURASI (bias + varians):
    RMS revisi = seberapa jauh estimasi real-time dari nilai final.
    Ini mencakup varians, sehingga tidak identik dengan uji (A).

TEMUAN PENTING
--------------
Kesimpulan (B) BERGANTUNG pada pilihan respons:
  - log_IHK      : SNR ~ 34   -> LL menang telak (bias mendominasi)
  - infl_pct_mtm : SNR ~ 0.3  -> NW menang (varians mendominasi; deru 3.4x sinyal)
Keunggulan teoretis LPR TIDAK otomatis terwujud pada data ber-SNR rendah.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
T = np.arange(1, len(df) + 1, dtype=float)
n = len(df)
H_GRID = [9, 12, 18, 24]          # bandwidth dalam BULAN
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


RESPONSES = [("log_IHK", df.log_IHK.values), ("infl_pct_mtm", df.infl_pct_mtm.values)]
all_res = {}

for name, Y in RESPONSES:
    fit = np.array([wvec(t, T, 12, 1) @ Y for t in T])
    resid = Y - fit
    sig = resid.std(); sig2 = sig**2
    snr = fit.std() / sig

    rows = []
    detail = {}
    for h in H_GRID:
        rec = {"h": h}
        b2, vn, vl, rn, rl, mps = [], [], [], [], [], []
        for t0 in range(T0_MIN, n - h + 1):
            w0 = wvec(t0, T[:t0], h, 0)
            w1 = wvec(t0, T[:t0], h, 1)
            if w0 is None or w1 is None:
                continue
            mp = slope_at(t0, T, Y, h)
            fin = wvec(t0, T, h, 1) @ Y
            mps.append(mp)
            b2.append((0.375 * h * mp) ** 2)
            vn.append(sig2 * np.sum(w0**2))
            vl.append(sig2 * np.sum(w1**2))
            rn.append(fin - w0 @ Y[:t0])
            rl.append(fin - w1 @ Y[:t0])
        mps, rn, rl = np.array(mps), np.array(rn), np.array(rl)
        rec["RMSrev_NW"] = np.sqrt(np.mean(rn**2))
        rec["RMSrev_LL"] = np.sqrt(np.mean(rl**2))
        rec["obs_MSE_NW/LL"] = rec["RMSrev_NW"]**2 / rec["RMSrev_LL"]**2
        rec["pred_MSE_NW/LL"] = (np.mean(b2) + np.mean(vn)) / np.mean(vl)
        rec["slope_NW"] = np.polyfit(mps, rn, 1)[0]
        rec["slope_teori"] = 0.375 * h
        rec["rasio_slope"] = rec["slope_NW"] / rec["slope_teori"]
        rec["corr_NW"] = np.corrcoef(mps, rn)[0, 1]
        rec["corr_LL"] = np.corrcoef(mps, rl)[0, 1]
        rows.append(rec)
        if h == 12:
            detail = {"mps": mps, "rn": rn, "rl": rl}

    res = pd.DataFrame(rows)
    all_res[name] = (res, snr, sig, detail)

    print("=" * 104)
    print(f"RESPONS: {name}   |   sd(deru)={sig:.5f}   sd(tren)={fit.std():.5f}   SNR={snr:.2f}")
    print("=" * 104)
    print("(A) UJI TEORI — regresi Revisi_NW terhadap m'(t0); prediksi kemiringan = 0.375*h")
    print(res[["h", "slope_NW", "slope_teori", "rasio_slope", "corr_NW", "corr_LL"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n(B) UJI AKURASI — RMS revisi (mencakup bias DAN varians)")
    print(res[["h", "RMSrev_NW", "RMSrev_LL", "obs_MSE_NW/LL", "pred_MSE_NW/LL"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"  -> {'LL UNGGUL' if res['obs_MSE_NW/LL'].mean() > 1 else 'NW UNGGUL'}"
          f" (rasio MSE NW/LL rata-rata = {res['obs_MSE_NW/LL'].mean():.3f})\n")

# ---------- plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, _) in zip(axes, RESPONSES):
    res, snr, sig, det = all_res[name]
    ax.scatter(det["mps"], det["rn"], s=14, color="#c0504d", alpha=.7, label="Revisi NW")
    ax.scatter(det["mps"], det["rl"], s=14, color="#2e7d32", alpha=.7, label="Revisi LL")
    xs = np.linspace(det["mps"].min(), det["mps"].max(), 50)
    ax.plot(xs, 0.375 * 12 * xs, "k--", lw=1.6, label="teori: 0.375·h·m'(t₀)")
    ax.axhline(0, color="grey", lw=.7)
    ax.set_xlabel("m'(t₀)  — kemiringan lokal")
    ax.set_ylabel("Revisi")
    ax.set_title(f"{name}   (SNR = {snr:.2f})", fontsize=11, weight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("Revision analysis, IHK BPS 2010–2026, h = 12 bulan\n"
             "Revisi NW mengikuti garis teori; revisi LL tak berpola (murni deru)", fontsize=12)
plt.tight_layout()
p = OUT_DIR / "revision_analysis.png"
plt.savefig(p, dpi=135)
print(f"[tersimpan] {p}")
