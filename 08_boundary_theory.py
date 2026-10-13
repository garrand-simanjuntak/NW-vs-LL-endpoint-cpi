"""
08_boundary_theory.py
Formula TERTUTUP bias batas + verifikasi terhadap bias eksak numerik.

TEORI
-----
Titik batas kanan diparametrisasi x = 1 - c*h, c in [0,1); c >= 1 = interior.
Momen kernel terpotong:      nu_j(c) = int_{-c}^{1} u^j K(u) du
(batas kanan: momen ganjil berganti tanda)

    Bias_NW(x) = h*m'(x) * nu1/nu0  +  (h^2/2)*m''(x) * nu2/nu0  + o(h^2)
    Bias_LL(x) = (h^2/2)*m''(x) * (nu2^2 - nu1*nu3)/(nu0*nu2 - nu1^2) + o(h^2)

Interior (c>=1): nu1 = nu3 = 0, nu0 = 1  =>  KEDUANYA = (h^2/2)*mu2*m''(x)
  -> menjelaskan temuan NW == LL persis di interior pada desain equidistant.
Batas   (c<1) : nu1 != 0  =>  NW punya suku O(h); LL tetap O(h^2).
  -> INILAH boundary effect dalam bentuk tertutup.

Epanechnikov K(u)=3/4(1-u^2), momen terpotong bentuk tertutup:
    nu0(c) = 3/4 (2/3 + c - c^3/3)
    nu1(c) = 3/16 (1 - c^2)^2            <- nol persis di c=1, maks 3/16 di c=0
    nu2(c) = 3/4 (2/15 + c^3/3 - c^5/5)
    nu3(c) = 3/4 (1/12 - c^4/4 + c^6/6)

Di titik batas ekstrem x=1 (c=0):  nu0=1/2, nu1=-3/16
    Bias_NW(1) ~ -0.375 * h * m'(1)

REZIM VALIDASI
--------------
Formula bersifat asimtotik dalam h->0 DAN nh->inf. Menguji dengan n tetap
sambil h->0 akan GAGAL: jendela kernel di batas hanya memuat ~nh titik,
sehingga aproksimasi Riemann S_j ~ n h^(j+1) nu_j runtuh. Uji yang benar
memakai limit ganda: h->0 dengan n ~ C/h^2 (sehingga nh = C/h -> inf).
"""
import numpy as np
from lpr_core import exact_bias
from sim_functions import TARGETS


def nu(j, c):
    """Momen Epanechnikov terpotong pada [-c, 1], bentuk tertutup."""
    if j == 0: return 0.75 * (2/3 + c - c**3/3)
    if j == 1: return (3/16) * (1 - c**2)**2
    if j == 2: return 0.75 * (2/15 + c**3/3 - c**5/5)
    if j == 3: return 0.75 * (1/12 - c**4/4 + c**6/6)
    raise ValueError(j)


def bias_theory(x, h, p, tgt, side="right"):
    """Bias teori bentuk tertutup di titik x."""
    c = min((1.0 - x)/h if side == "right" else x/h, 1.0)
    s = -1.0 if side == "right" else 1.0
    n0, n1, n2, n3 = nu(0, c), s*nu(1, c), nu(2, c), s*nu(3, c)
    m1 = float(tgt.dm(np.array([x]))[0])
    m2 = float(tgt.d2m(np.array([x]))[0])
    if p == 0:
        return h*m1*n1/n0 + 0.5*h**2*m2*n2/n0
    return 0.5*h**2*m2*(n2**2 - n1*n3)/(n0*n2 - n1**2)


if __name__ == "__main__":
    KERNEL = "epanechnikov"
    n0, n1, n2, n3 = nu(0, 0), -nu(1, 0), nu(2, 0), -nu(3, 0)
    COEF_NW = n1/n0
    COEF_LL = (n2**2 - n1*n3)/(n0*n2 - n1**2)

    print("=" * 92)
    print("KOEFISIEN BIAS DI TITIK BATAS EKSTREM x = 1  (c = 0)")
    print("=" * 92)
    print(f"  NW : nu1/nu0                       = {COEF_NW:.6f}   (= -3/8)")
    print(f"  LL : (nu2^2-nu1*nu3)/(nu0*nu2-nu1^2) = {COEF_LL:.6f}")
    print(f"  LL interior (mu2)                  = {0.2:.6f}")
    print(f"  -> Bias_NW(1) ~ {COEF_NW:.3f} * h * m'(1)      [O(h)]")
    print(f"  -> Bias_LL(1) ~ {0.5*COEF_LL:.4f} * h^2 * m''(1)   [O(h^2)]")

    tgt = TARGETS[1]  # M2_exp
    m1 = float(tgt.dm(np.array([1.0]))[0])
    m2 = float(tgt.d2m(np.array([1.0]))[0])

    print()
    print("=" * 92)
    print("VALIDASI — limit ganda: h -> 0 dengan n = ceil(300/h^2), jadi nh = 300/h -> inf")
    print("Uji: bias_eksak / suku_pemimpin_teori  harus -> 1")
    print("=" * 92)
    print(f"{'h':>9} {'n':>9} {'nh':>7} {'NW_eksak':>11} {'NW_teori':>11} {'rasio':>8}"
          f" {'LL_eksak':>12} {'LL_teori':>12} {'rasio':>8}")
    for h in [0.20, 0.10, 0.05, 0.025, 0.0125, 0.00625]:
        n = int(np.ceil(300/h**2))
        X = np.arange(1, n+1)/n
        be_nw = exact_bias(np.array([1.0]), X, h, 0, tgt, KERNEL)[0]
        be_ll = exact_bias(np.array([1.0]), X, h, 1, tgt, KERNEL)[0]
        lead_nw = h*m1*COEF_NW
        lead_ll = 0.5*h**2*m2*COEF_LL
        print(f"{h:9.5f} {n:9d} {n*h:7.0f} {be_nw:11.6f} {lead_nw:11.6f} {be_nw/lead_nw:8.4f}"
              f" {be_ll:12.6f} {lead_ll:12.6f} {be_ll/lead_ll:8.4f}")

    print()
    print("=" * 92)
    print("FAKTOR KONSTANTA BIAS LL SEPANJANG WILAYAH BATAS")
    print("=" * 92)
    print(f"{'c':>6} {'faktor_LL':>12} {'rasio thd interior (mu2=0.2)':>30}")
    for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        a0, a1, a2, a3 = nu(0, c), -nu(1, c), nu(2, c), -nu(3, c)
        fac = (a2**2 - a1*a3)/(a0*a2 - a1**2)
        print(f"{c:6.2f} {fac:12.6f} {fac/0.2:30.4f}")
