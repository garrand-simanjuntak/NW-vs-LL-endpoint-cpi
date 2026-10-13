"""
15_seasonal_sensitivity.py
Sensitivitas terhadap musiman (Reviewer 1, komentar Data): apakah musiman
menggerakkan SNR inflasi bulanan (0.29), penolakan AR(1) pada log IHK, dan
pembalikan Test (B)?

VARIAN
  S0  mentah (seperti naskah)
  S1  r_t dikurangi faktor bulan kalender deterministik (12 dummy, jumlah nol)
  S2  S1 + dummy bulan Idulfitri dan bulan sebelumnya (Ramadan bergeser
      sekitar 11 hari per tahun, sehingga tidak tertangkap dummy kalender)
Faktor ditaksir dengan OLS pada sampel penuh. Ini bukan penyesuaian musiman
gaya X-13/STL (tidak ada filter satu sisi di ujung deret), sehingga argumen
sirkularitas di Bagian 2.8 tidak berlaku. log IHK direkonstruksi ulang dari
r_t yang disesuaikan dengan Eq. (20).

UKURAN PER VARIAN
  * ACF residual (lag 1, 2, 3, 12) setelah fit LL h = 12
  * SNR naskah, lantai noise iid (SNR noise murni), dan SNR terkoreksi bias
      var_sinyal = var(fit) - sigma2 * tr(C L L' C)/n,
      sigma2     = RSS / tr((I-L)'(I-L))
  * autokorelasi lag-1 differensi kedua log IHK (naskah: -0.1233)
  * CV/GCV pada log IHK, grid 6..48 bulan
  * Test (A) rasio terhadap prediksi diskret dan Test (B) h bersama

TANGGAL IDULFITRI: tanggal resmi pemerintah RI; verifikasi dengan SKB
cuti bersama sebelum dilaporkan.
Keluaran: output/seasonal_sensitivity.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
n = len(df)
T = np.arange(1, n + 1, dtype=float)
H_GRID = [9, 12, 18, 24]
T0_MIN = 48

EID = {2010: (9, 10), 2011: (8, 30), 2012: (8, 19), 2013: (8, 8), 2014: (7, 28),
       2015: (7, 17), 2016: (7, 6), 2017: (6, 25), 2018: (6, 15), 2019: (6, 5),
       2020: (5, 24), 2021: (5, 13), 2022: (5, 2), 2023: (4, 22), 2024: (4, 10),
       2025: (3, 31), 2026: (3, 20)}


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def lp_rows(t0, Tv, h, p):
    w = epan((Tv - t0) / h)
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return np.linalg.inv(ZtW @ Z) @ ZtW


def smoother(h, p=1):
    return np.vstack([lp_rows(t, T, h, p)[0] for t in T])


def seasonal_adjust(r, with_eid):
    month = df.date.dt.month.values
    D = np.column_stack([(month == m).astype(float) for m in range(1, 13)])
    D = D - D.mean(axis=0)                     # kontras jumlah nol, rata-rata r dipertahankan
    cols = [D]
    if with_eid:
        eid_m = np.array([1.0 if (y, m) == (y, EID[y][0]) else 0.0
                          for y, m in zip(df.date.dt.year, month)])
        pre_m = np.roll(eid_m, -1); pre_m[-1] = 0.0
        E = np.column_stack([eid_m, pre_m])
        cols.append(E - E.mean(axis=0))
    Xd = np.column_stack([np.ones(n)] + cols)
    beta = np.linalg.lstsq(Xd, r, rcond=None)[0]
    return r - Xd[:, 1:] @ beta[1:]


def acf(e, lags):
    e = e - e.mean()
    return [np.corrcoef(e[k:], e[:-k])[0, 1] for k in lags]


L12 = smoother(12)
I = np.eye(n); C = I - np.ones((n, n)) / n
TR_L = np.trace(C @ L12 @ L12.T @ C)
TR_R = np.trace(C @ (I - L12) @ (I - L12).T @ C)
FLOOR = np.sqrt(TR_L / TR_R)
DOF = np.trace((I - L12).T @ (I - L12))


def snr_block(Y):
    fit = L12 @ Y
    res = Y - fit
    snr = fit.std() / res.std()
    s2 = (res @ res) / DOF
    vs = fit.var() - s2 * TR_L / n
    return snr, np.sqrt(max(vs, 0.0)) / np.sqrt(s2), res


def cv_pick(Y, grid):
    cv, gcv = [], []
    for h in grid:
        L = smoother(h)
        e = Y - L @ Y
        cv.append(np.mean((e / (1 - np.diag(L))) ** 2))
        gcv.append(np.mean(e**2) / (1 - np.trace(L) / n) ** 2)
    return grid[int(np.argmin(cv))], grid[int(np.argmin(gcv))], bool(np.all(np.diff(cv) > 0))


def revision_tests(Y):
    out = {}
    for h in H_GRID:
        slope, rn, rl = [], [], []
        for t0 in range(T0_MIN, n - h + 1):
            fin = lp_rows(t0, T, h, 1)[0] @ Y
            rn.append(fin - lp_rows(t0, T[:t0], h, 0)[0] @ Y[:t0])
            rl.append(fin - lp_rows(t0, T[:t0], h, 1)[0] @ Y[:t0])
            slope.append(lp_rows(t0, T, h, 1)[1] @ Y)
        k = np.arange(h); K = 1 - (k / h) ** 2
        disc = (K * k).sum() / K.sum()
        rn, rl = np.array(rn), np.array(rl)
        out[h] = (np.polyfit(slope, rn, 1)[0] / disc, np.mean(rn**2) / np.mean(rl**2))
    return out


r_raw = df.infl_pct_mtm.values
VARIANTS = {"S0_raw": r_raw,
            "S1_month": seasonal_adjust(r_raw, False),
            "S2_month+eid": seasonal_adjust(r_raw, True)}
CV_GRID = np.arange(6, 49, dtype=float)

rows = []
for vname, r in VARIANTS.items():
    logc = np.log(100 * np.cumprod(1 + r / 100))
    d2 = np.diff(logc, 2)
    for resp, Y in [("log_IHK", logc), ("infl_pct_mtm", r)]:
        snr, snr_c, res = snr_block(Y)
        rec = {"variant": vname, "response": resp, "SNR": snr, "SNR_floor_iid": FLOOR,
               "SNR_over_floor": snr / FLOOR, "SNR_biascorr": snr_c}
        for k, a in zip([1, 2, 3, 12], acf(res, [1, 2, 3, 12])):
            rec[f"acf{k}"] = a
        if resp == "log_IHK":
            rec["d2_acf1"] = acf(d2, [1])[0]
            rec["CV_h"], rec["GCV_h"], rec["CV_monotone"] = cv_pick(Y, CV_GRID)
        for h, (ra, rb) in revision_tests(Y).items():
            rec[f"A_ratio_disc_h{h}"] = ra
            rec[f"B_MSE_NW/LL_h{h}"] = rb
        rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "seasonal_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
f3 = lambda v: f"{v:.3f}"
print(f"Lantai noise iid SNR (LL h=12, n={n}) = {FLOOR:.3f}\n")
print("SNR dan ACF residual")
print(res[["variant", "response", "SNR", "SNR_over_floor", "SNR_biascorr",
           "acf1", "acf2", "acf3", "acf12"]].to_string(index=False, float_format=f3))
print("\nlog IHK: differensi kedua dan seleksi bandwidth (AR(1) mensyaratkan d2_acf1 di [-0.667, -0.501])")
print(res[res.response == "log_IHK"][["variant", "d2_acf1", "CV_h", "GCV_h", "CV_monotone"]]
      .to_string(index=False, float_format=f3))
print("\nTest (A) rasio terhadap diskret dan Test (B) rasio MSE NW/LL (h bersama)")
cols = ["variant", "response"] + [c for c in res.columns if c.startswith(("A_", "B_"))]
print(res[cols].to_string(index=False, float_format=f3))
print(f"\n[tersimpan] {OUT_DIR / 'seasonal_sensitivity.csv'}")
