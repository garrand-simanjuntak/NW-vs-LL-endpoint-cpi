"""
13_bandwidth_real.py
Menerapkan CV, GCV, dan Modified CV pada data IHK riil (log_IHK), dan menguji
apakah galat mengikuti AR(1) lewat estimator berbasis selisih kedua.

CATATAN REPRODUKTIBILITAS: angka-angka ini SEBELUMNYA dilaporkan di
RINGKASAN_HASIL.md tapi dihitung lewat kode ad-hoc yang tidak disimpan --
celah yang ditemukan saat audit draft. Skrip ini menutup celah tersebut
dengan menyimpan versi yang dapat dijalankan ulang.

Estimator rho berbasis selisih kedua (Rice 1984 / Gasser et al. 1986):
    d2_t = Y_t - 2*Y_{t-1} + Y_{t-2}
Untuk galat AR(1) dengan varians marginal sigma^2:
    Var(d2)            = sigma^2 * (6 - 8*rho + 2*rho^2)
    Cov(d2_t, d2_{t+1}) = sigma^2 * (-4 + 7*rho - 4*rho^2 + rho^3)
    r1_teori(rho)       = Cov / Var
rho_hat dicari dengan menyamakan r1_teori(rho) = r1 teramati dari data.
Jika tidak ada akar valid di (0,1), model AR(1) DITOLAK oleh data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv")
Y = df.log_IHK.values
T = np.arange(1, len(df) + 1, dtype=float)
n = len(df)
P = 1                              # Local Linear
H_GRID = np.arange(6, 49, 1).astype(float)


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


# ---------------------------------------------------------------
# BAGIAN A — uji apakah galat IHK mengikuti AR(1)
# ---------------------------------------------------------------
print("=" * 78)
print("UJI AR(1) via estimator berbasis selisih kedua (Rice/Gasser)")
print("=" * 78)

d2 = Y[2:] - 2 * Y[1:-1] + Y[:-2]
r1_obs = np.corrcoef(d2[:-1], d2[1:])[0, 1]


def r1_theory(rho):
    var = 6 - 8 * rho + 2 * rho**2
    cov = -4 + 7 * rho - 4 * rho**2 + rho**3
    return cov / var


rng_lo, rng_hi = r1_theory(0.0), r1_theory(0.95)
print(f"r1 selisih-kedua TERAMATI      : {r1_obs:+.4f}")
print(f"rentang teoretis AR(1), rho in [0, 0.95]: [{rng_lo:+.4f}, {rng_hi:+.4f}]")

try:
    rho_hat = brentq(lambda r: r1_theory(r) - r1_obs, 0.0, 0.95)
    ar1_valid = True
    print(f"rho_hat  = {rho_hat:.4f}  -> model AR(1) COCOK")
except ValueError:
    rho_hat = np.nan
    ar1_valid = False
    print("Tidak ada akar valid di (0, 0.95) -> TERAMATI DI LUAR RENTANG")
    print("=> model AR(1) DITOLAK oleh data. Kemungkinan: MA negatif,")
    print("   musiman sisa, atau heteroskedastisitas.")

# ---------------------------------------------------------------
# BAGIAN B — CV, GCV, MCV pada data riil
# ---------------------------------------------------------------
print()
print("=" * 78)
print("BANDWIDTH TERPILIH — data IHK riil (log_IHK, Local Linear)")
print("=" * 78)

idx = np.arange(n)
ELLS = [0, 2, 4, 8]     # l=0 setara LOOCV; l=2,4,8 dipertahankan utk sensitivitas
                        # (aturan l* = ln(0.1)/ln(rho) TIDAK berlaku -- AR(1) ditolak)

rows = []
for h in H_GRID:
    L = np.vstack([wrow(t, T, h, P) for t in T])
    fit = L @ Y
    e = Y - fit
    lii = np.diag(L)
    rec = {
        "h": h,
        "CV": np.mean((e / (1 - lii)) ** 2),
        "GCV": np.mean(e**2) / (1 - np.trace(L) / n) ** 2,
    }
    for ell in ELLS:
        tot = 0.0
        cnt = 0
        for i in range(n):
            mask = np.abs(idx - i) > ell
            w = wrow(T[i], T[mask], h, P)
            if w is None:
                continue
            tot += (Y[i] - w @ Y[mask]) ** 2
            cnt += 1
        rec[f"MCV_l{ell}"] = tot / cnt if cnt == n else np.nan
    rows.append(rec)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)

print(f"\n{'kriteria':>12} {'h* (bulan)':>12}")
sel = {}
for c in ["CV", "GCV"] + [f"MCV_l{e}" for e in ELLS]:
    if res[c].notna().sum() == 0:
        continue
    hh = res.loc[res[c].idxmin(), "h"]
    sel[c] = hh
    print(f"{c:>12} {hh:12.0f}")

res.to_csv(OUT_DIR / "bandwidth_real.csv", index=False)
print(f"\n[tersimpan] {OUT_DIR/'bandwidth_real.csv'}")

print()
print("=" * 78)
print("RINGKASAN UNTUK RINGKASAN_HASIL.md")
print("=" * 78)
print(f"r1_obs = {r1_obs:.4f} | rentang AR(1) = [{rng_lo:.4f}, {rng_hi:.4f}] "
      f"| AR(1) valid? {ar1_valid}")
for c, hh in sel.items():
    print(f"h*_{c} = {hh:.0f} bulan")
