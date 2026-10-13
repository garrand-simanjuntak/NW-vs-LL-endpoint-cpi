"""
16_testB_own_bandwidth.py
Test (B) dengan bandwidth masing-masing estimator (Reviewer 1, komentar
Desain Revisi: "compare each estimator at its own selected or optimal
bandwidth").

MASALAH PADA DESAIN LAMA
Revisi = m_final(t0; h) - m_rt(t0; h) memakai h yang sama untuk final dan
real-time. Jika h dibedakan antar estimator, proxy kebenarannya ikut berbeda,
dan RMS revisi degenerate saat h -> 0 (final dan real-time sama-sama
menginterpolasi data).

DESAIN BARU
Satu estimasi final referensi bersama m_ref(t0) = LL interior pada h_ref.
Karena NW == LL di interior, referensi ini netral terhadap kedua pesaing.
Estimasi real-time NW dan LL dihitung pada grid h sendiri-sendiri:
    D_e(h) = mean_t0 [ m_ref(t0) - m_rt,e(t0; h) ]^2,   e in {NW, LL}.
Karena m_ref tidak bergantung pada h, D_e(h) TIDAK degenerate saat h -> 0
(estimasi real-time ber-h kecil mengikuti noise dan menjauh dari referensi).
Dilaporkan:
  * in-sample : min_h D_e(h), h*_e, dan rasio D_NW* / D_LL*
  * out-of-sample : h*_e dipilih pada t0 <= 120, dievaluasi pada t0 > 120
  * sensitivitas h_ref in {12, 18, 24} dan varian musiman S0 / S2.
Satuan RMS: log poin untuk log IHK, poin persen untuk inflasi bulanan.
Keluaran: output/testB_own_bandwidth.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / "ihk_bps_2010_2026.csv", parse_dates=["date"])
n = len(df)
T = np.arange(1, n + 1, dtype=float)
H_RT = np.arange(2, 49)          # h = 2: NW memakai 2 titik terakhir; LL butuh >= 3 titik
H_REF = [12, 18, 24]
T0_MIN, T0_SPLIT = 48, 120

EID_MONTH = {2010: 9, 2011: 8, 2012: 8, 2013: 8, 2014: 7, 2015: 7, 2016: 7, 2017: 6,
             2018: 6, 2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3, 2026: 3}


def epan(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def lp0(t0, Tv, h, p):
    w = epan((Tv - t0) / h)
    if (w > 0).sum() < p + 2:                  # aturan kelayakan yang sama dengan skrip 10
        return None
    Z = np.vander(Tv - t0, N=p + 1, increasing=True)
    ZtW = Z.T * w
    return (np.linalg.inv(ZtW @ Z) @ ZtW)[0]


def adjust_s2(r):
    month = df.date.dt.month.values
    D = np.column_stack([(month == m).astype(float) for m in range(1, 13)])
    eid = np.array([1.0 if m == EID_MONTH[y] else 0.0 for y, m in zip(df.date.dt.year, month)])
    pre = np.roll(eid, -1); pre[-1] = 0.0
    S = np.column_stack([D, eid, pre]); S = S - S.mean(axis=0)
    Xd = np.column_stack([np.ones(n), S])
    b = np.linalg.lstsq(Xd, r, rcond=None)[0]
    return r - S @ b[1:]


r0 = df.infl_pct_mtm.values
SERIES = {}
for vname, r in [("S0_raw", r0), ("S2_month+eid", adjust_s2(r0))]:
    SERIES[(vname, "log_IHK")] = np.log(100 * np.cumprod(1 + r / 100))
    SERIES[(vname, "infl_pct_mtm")] = r

# real-time weights tidak bergantung pada Y: hitung sekali
T0_ALL = np.arange(T0_MIN, n - min(H_REF) + 1)
RT = {(p, h, t0): lp0(t0, T[:t0], h, p) for p in (0, 1) for h in H_RT for t0 in T0_ALL}

rows = []
for (vname, resp), Y in SERIES.items():
    for h_ref in H_REF:
        t0s = np.arange(T0_MIN, n - h_ref + 1)
        ref = np.array([lp0(t0, T, h_ref, 1) @ Y for t0 in t0s])
        D = {}
        for p in (0, 1):
            D[p] = np.array([[np.inf if RT[(p, h, t0)] is None else
                              (ref[i] - RT[(p, h, t0)] @ Y[:t0]) ** 2
                              for i, t0 in enumerate(t0s)] for h in H_RT])   # (H, t0)
        early, late = t0s <= T0_SPLIT, t0s > T0_SPLIT
        rec = {"variant": vname, "response": resp, "h_ref": h_ref, "n_t0": len(t0s)}
        for p, lab in [(0, "NW"), (1, "LL")]:
            m_all = D[p].mean(axis=1)
            i_star = int(np.argmin(m_all))
            i_tr = int(np.argmin(D[p][:, early].mean(axis=1)))
            rec[f"h_{lab}"] = H_RT[i_star]
            rec[f"RMS_{lab}"] = np.sqrt(m_all[i_star])
            rec[f"h_{lab}_train"] = H_RT[i_tr]
            rec[f"RMS_{lab}_oos"] = np.sqrt(D[p][i_tr, late].mean())
            feas = np.isfinite(m_all)
            rec[f"h_{lab}_at_edge"] = H_RT[i_star] in (H_RT[feas].min(), H_RT[feas].max())
        rec["MSE_NW/LL"] = rec["RMS_NW"] ** 2 / rec["RMS_LL"] ** 2
        rec["MSE_NW/LL_oos"] = rec["RMS_NW_oos"] ** 2 / rec["RMS_LL_oos"] ** 2
        rec["preferred"] = "LL" if rec["MSE_NW/LL"] > 1 else "NW"
        rec["preferred_oos"] = "LL" if rec["MSE_NW/LL_oos"] > 1 else "NW"
        rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "testB_own_bandwidth.csv", index=False)
pd.set_option("display.width", 250)
print("TEST (B) DENGAN BANDWIDTH MASING-MASING, REFERENSI FINAL BERSAMA")
print(res[["variant", "response", "h_ref", "n_t0", "h_NW", "RMS_NW", "h_LL", "RMS_LL",
           "MSE_NW/LL", "preferred", "h_NW_train", "h_LL_train", "MSE_NW/LL_oos",
           "preferred_oos", "h_NW_at_edge", "h_LL_at_edge"]]
      .to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print(f"\n[tersimpan] {OUT_DIR / 'testB_own_bandwidth.csv'}")
