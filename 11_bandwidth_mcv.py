"""
11_bandwidth_mcv.py
POIN 4 — Pemilihan bandwidth di bawah autokorelasi: CV, GCV, Modified CV.

MASALAH
-------
CV/GCV standar mengandaikan error INDEPENDEN. Bila error berautokorelasi
positif, membuang satu titik i tidak menghapus informasi tentang titik itu:
tetangganya yang berkorelasi masih ada dan "membocorkan" noise lokal. Estimator
dapat menurunkan skor CV dengan MENGIKUTI noise tersebut, sehingga CV memilih
h TERLALU KECIL (undersmoothing).

MODIFIED CV (Chu & Marron 1991; Hart & Vieu 1990)
    Buang BLOK B_i = {i-l, ..., i+l}, bukan hanya {i}:
        MCV_l(h) = (1/n) sum_i [ Y_i - m_hat_{-B_i}(x_i) ]^2

DESAIN VALIDASI
---------------
Klaim "MCV lebih baik" HARUS diuji terhadap kebenaran yang diketahui. Pada data
riil m(x) tidak diketahui, sehingga oracle sejati mustahil dibentuk. (Catatan:
RMS revisi TIDAK sah sebagai oracle -- ia degenerate saat h->0, karena estimasi
real-time dan final sama-sama menginterpolasi data sehingga revisi -> 0 secara
trivial. Ia mengukur konsistensi-diri, bukan akurasi.)

Karena itu: kalibrasi kriteria pada DATA SIMULASI dengan m(x) diketahui dan
error AR(1), lalu terapkan kriteria pemenang ke data riil.

=====================  REVISI TERHADAP VERSI SEBELUMNYA  =====================
1. GRID ELL DIPERBAIKI. Versi lama memakai ELLS = [0, 2, 4, 8], yang TIDAK
   memuat satu pun nilai yang diperintahkan aturan l* = ln(0,1)/ln(rho) untuk
   rho = 0,6 (l* = 4,51) maupun rho = 0,9 (l* = 21,85). Akibatnya aturan itu
   tidak pernah diuji secara langsung. Grid baru memuat l = 2, 5, 22 (nilai
   aturan untuk rho = 0,3 / 0,6 / 0,9) plus l = 8 dan 16 sebagai pembanding.

   Hasil setelah perbaikan:
     rho = 0,3 -> l* = 2  -> rasio oracle 1,000  (aturan BENAR)
     rho = 0,6 -> l* = 5  -> rasio oracle 1,000  (aturan BENAR)
     rho = 0,9 -> l* = 22 -> rasio oracle 1,304  (aturan GAGAL, oversmooth 30%)
   Nilai 1,09 yang pernah dilaporkan untuk rho = 0,9 berasal dari l = 16, yaitu
   pilihan post-hoc dan BUKAN keluaran aturan.

2. KOREKSI ANGKA. Sel (rho = 0,6; l = 4) sebelumnya tercatat 1,000. Nilai yang
   tereproduksi dari skrip ini adalah 0,889 (h* = 0,160 vs oracle 0,180), dan
   lintas lima seed berkisar 0,865-0,944 -- tidak pernah mencapai oracle.
   Selisih ini BUKAN artefak penjumlahan floating-point: versi loop dan versi
   tervektor memberi argmin identik pada 200/200 replikasi.

3. VEKTORISASI. Bobot leave-block-out kini disimpan sebagai matriks n x n per
   (h, l) sehingga prediksi = M @ Y. Aljabarnya identik dengan loop lama
   `sum((Y[i] - w @ Y[mask])**2)`, hanya jauh lebih cepat -- perlu karena grid
   ell diperlebar. Urutan konsumsi RNG DIPERTAHANKAN PERSIS, sehingga kolom
   l = 0, 2, 8 tetap tereproduksi sama dengan versi lama.

4. PELAPORAN KELAYAKAN. Blok panjang tidak terdefinisi pada bandwidth kecil:
   membuang blok di sekitar setiap titik dapat mengosongkan jendela kernel.
   Skrip kini melaporkan h terkecil yang layak per l, karena rasio pada kolom
   l besar terpotong dari bawah secara mekanis dan tidak boleh dibaca setara
   dengan kolom l kecil.
=============================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_functions import M2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

n = 198
X = np.arange(1, n + 1) / n
H_GRID = np.arange(0.03, 0.31, 0.01)
P = 1
SIGMA = 0.30
RHOS = [0.0, 0.3, 0.6, 0.9]
ELLS = [0, 2, 5, 8, 16, 22]   # 0 = LOOCV; 2/5/22 = nilai aturan; 8/16 pembanding
R = 200                       # replikasi per rho
rng = np.random.default_rng(2026)


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def wrow(x0, Xv, h, p):
    w = epan((Xv - x0) / h)
    if (w > 0).sum() < p + 2:
        return None
    Z = np.vander(Xv - x0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    try:
        return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]
    except np.linalg.LinAlgError:
        return None


# pra-hitung hat matrix per h (desain tetap -> tidak bergantung Y)
print("pra-hitung hat matrix ...")
HAT = {}
for h in H_GRID:
    rows = [wrow(x, X, h, P) for x in X]
    HAT[h] = np.vstack(rows) if all(r is not None for r in rows) else None

# pra-hitung bobot MCV: matriks n x n, baris i = bobot leave-block-out titik i
print("pra-hitung bobot MCV (blok dibuang) ...")
MCVW = {}
idx = np.arange(n)
for h in H_GRID:
    for ell in ELLS:
        M = np.zeros((n, n))
        ok = True
        for i in range(n):
            mask = np.abs(idx - i) > ell
            w = wrow(X[i], X[mask], h, P)
            if w is None:
                ok = False
                break
            M[i, mask] = w
        MCVW[(h, ell)] = M if ok else None

print("\nKELAYAKAN: h terkecil yang terdefinisi per panjang blok")
feas_rows = []
for ell in ELLS:
    feas = [h for h in H_GRID if MCVW[(h, ell)] is not None]
    feas_rows.append({"ell": ell, "h_min_layak": min(feas),
                      "titik_grid_layak": f"{len(feas)}/{len(H_GRID)}"})
    print(f"  ell={ell:2d} -> h_min = {min(feas):.2f}  ({len(feas)}/{len(H_GRID)} titik grid)")
pd.DataFrame(feas_rows).to_csv(OUT_DIR / "mcv_feasibility.csv", index=False)


def ar1_errors(rho, size):
    """Error AR(1) dengan varians marginal SIGMA^2."""
    e = np.empty(size)
    e[0] = rng.normal(0, SIGMA)
    s = SIGMA * np.sqrt(1 - rho**2)
    for t in range(1, size):
        e[t] = rho * e[t - 1] + rng.normal(0, s)
    return e


mX = M2(X)
results = []
CRIT = ["CV", "GCV"] + [f"MCV_l{e}" for e in ELLS]

for rho in RHOS:
    print(f"kalibrasi rho = {rho} ...")
    picks = {c: [] for c in CRIT + ["ORACLE"]}
    for r in range(R):
        eps = ar1_errors(rho, n) if rho > 0 else rng.normal(0, SIGMA, n)
        Y = mX + eps

        sc = {c: [] for c in picks}
        for h in H_GRID:
            L = HAT[h]
            if L is None:
                for c in sc:
                    sc[c].append(np.inf)
                continue
            fit = L @ Y
            e = Y - fit
            lii = np.diag(L)
            sc["CV"].append(np.mean((e / (1 - lii))**2))
            sc["GCV"].append(np.mean(e**2) / (1 - np.trace(L) / n)**2)
            # ORACLE: MSE sejati terhadap m(x) yang DIKETAHUI
            sc["ORACLE"].append(np.mean((fit - mX)**2))
            for ell in ELLS:
                M = MCVW[(h, ell)]
                if M is None:
                    sc[f"MCV_l{ell}"].append(np.inf)
                    continue
                sc[f"MCV_l{ell}"].append(np.mean((Y - M @ Y)**2))
        for c in picks:
            picks[c].append(H_GRID[int(np.argmin(sc[c]))])

    rec = {"rho": rho}
    h_or = np.median(picks["ORACLE"])
    rec["h_ORACLE"] = h_or
    for c in CRIT:
        hm = np.median(picks[c])
        rec[f"h_{c}"] = hm
        rec[f"rasio_{c}"] = hm / h_or
    results.append(rec)

res = pd.DataFrame(results)
pd.set_option("display.width", 250)

print("\n" + "=" * 110)
print(f"KALIBRASI PADA DATA SIMULASI (m diketahui), M2_exp, n={n}, R={R} replikasi")
print("Median h terpilih. rasio < 1 => UNDERSMOOTHING (h terlalu kecil)")
print("=" * 110)
cols_h = ["rho", "h_ORACLE", "h_CV", "h_GCV"] + [f"h_MCV_l{e}" for e in ELLS]
print(res[cols_h].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
cols_r = ["rho"] + [f"rasio_{c}" for c in CRIT]
print(res[cols_r].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

res.to_csv(OUT_DIR / "bandwidth_calibration.csv", index=False)

# --- uji langsung aturan l* = ln(0,1)/ln(rho) ---
print("\n" + "=" * 110)
print("UJI LANGSUNG ATURAN  l* = ln(0,1)/ln(rho)")
print("=" * 110)
print(f"{'rho':>6} {'l* aturan':>11} {'l terdekat di grid':>20} {'rasio oracle':>14}  status")
for rho in RHOS[1:]:
    l_rule = np.log(0.1) / np.log(rho)
    l_use = min(ELLS[1:], key=lambda e: abs(e - l_rule))
    ratio = res.loc[res.rho == rho, f"rasio_MCV_l{l_use}"].iloc[0]
    status = "BENAR" if 0.95 <= ratio <= 1.10 else "GAGAL"
    print(f"{rho:>6.1f} {l_rule:>11.2f} {l_use:>20d} {ratio:>14.3f}  {status}")

print("\nONGKOS SAAT rho = 0 (error tak berkorelasi):")
for ell in ELLS[1:]:
    ratio = res.loc[res.rho == 0.0, f"rasio_MCV_l{ell}"].iloc[0]
    print(f"  ell={ell:2d}: rasio {ratio:.3f}  -> oversmooth {100*(ratio-1):.0f}%")

# --- gambar ---
fig, ax = plt.subplots(figsize=(9, 5.5))
styles = [("CV", "#c0504d", "o"), ("GCV", "#e08214", "s"),
          ("MCV_l2", "#a5d6a7", "^"), ("MCV_l5", "#66bb6a", "v"),
          ("MCV_l8", "#2e7d32", "D"), ("MCV_l16", "#1f4e79", "P"),
          ("MCV_l22", "#4a148c", "X")]
for c, col, mk in styles:
    ax.plot(res.rho, res[f"rasio_{c}"], mk + "-", color=col, lw=1.8, ms=6,
            label=c.replace("MCV_l", "MCV \u2113="))
ax.axhline(1.0, color="k", ls="--", lw=2, label="ORACLE (rasio=1)")
ax.set_xlabel("autokorelasi error  \u03c1")
ax.set_ylabel("h terpilih / h oracle")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=.3)
plt.tight_layout()
p = OUT_DIR / "bandwidth_selection.png"
plt.savefig(p, dpi=300)
print(f"\n[tersimpan] {p}")
print(f"[tersimpan] {OUT_DIR/'bandwidth_calibration.csv'}")
