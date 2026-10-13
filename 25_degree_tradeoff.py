"""
25_degree_tradeoff.py
Revisi putaran 2 (Reviewer 2), DG-5: apakah trade-off bias batas dan varians real-time meluas ke derajat
polinomial yang lebih tinggi (lokal kuadratik, lokal kubik).

Desain sama dengan Bagian 3.7 (DG-4): estimasi real-time di observasi terakhir, h = 12 bulan, kernel
Epanechnikov K(k/h), jendela satu sisi 12 titik (lag k = 0, ..., 11). Bobot eksak dari lpr_core.lp_weights.

Keluaran per derajat p = 0, 1, 2, 3:
  - bobot pada observasi terkini w_0 (leverage, istilah Proietti dan Luati 2008);
  - faktor varians terhadap rata-rata trailing 12 bulan, galat i.i.d.: 12 * sum(w^2);
  - lag rata-rata sum(k * w_k) (nol untuk p >= 1 karena bobot mereproduksi tren linear);
  - bobot terkecil (negatif untuk p >= 1).
Pembanding dari literatur (bukan dihitung di sini): Proietti dan Luati (2008), Tabel 1, h = 6 (13 suku),
q = 0: w_0 = 0.2457 (d = 0), 0.5856 (d = 1), 0.9552 (d = 3), kernel Henderson.

Jalankan: python 25_degree_tradeoff.py > logs/25_degree_tradeoff.log
"""
import numpy as np
from lpr_core import lp_weights

H = 12
lags = np.arange(H)                      # k = 0 adalah observasi terkini
t = -lags.astype(float)                  # posisi relatif terhadap t0 (masa lalu negatif)

print(f"DG-5: estimasi real-time, h = {H} bulan, Epanechnikov, jendela satu sisi {H} titik")
print("  p  w_0(terkini)  faktor_varians_vs_rata2_trailing  lag_rata2  bobot_min")
for p in range(4):
    w = lp_weights(0.0, t, H, p)         # K((t - 0)/H): titik di lag 0..11 berbobot positif
    var_factor = H * np.sum(w ** 2)
    lag = np.sum(lags * w)
    print(f"  {p}  {w[0]:12.4f}  {var_factor:32.4f}  {lag:9.4f}  {w.min():9.4f}")

w_nw, w_ll, w_cub = (lp_weights(0.0, t, H, p) for p in (0, 1, 3))
print(f"\n  rasio varians kubik/LL = {np.sum(w_cub ** 2) / np.sum(w_ll ** 2):.3f}; "
      f"LL/NW = {np.sum(w_ll ** 2) / np.sum(w_nw ** 2):.3f}")
print("  cek: faktor NW dan LL harus sama dengan 24_worked_example.log (1.15 dan 3.79)")
