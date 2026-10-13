"""
lpr_core.py
Inti estimator: Nadaraya-Watson (p=0) dan Local Polynomial (p>=1),
dinyatakan dalam kerangka linear smoother yang sama.

    m_hat(x) = sum_t w_t(x) * Y_t

sehingga bias dapat dihitung EKSAK pada desain tetap:

    Bias(x) = sum_t w_t(x) * m(x_t) - m(x)      [deterministik]
"""
import numpy as np


# ---------- Kernel ----------
def epanechnikov(u):
    """K(u) = 3/4 (1-u^2) 1{|u|<=1}. Support kompak -> efek batas tegas."""
    return np.where(np.abs(u) <= 1, 0.75 * (1.0 - u**2), 0.0)


def gaussian(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


KERNELS = {"epanechnikov": epanechnikov, "gaussian": gaussian}


# ---------- Bobot ekuivalen (equivalent kernel) ----------
def lp_weights(x0, X, h, p, kernel="epanechnikov"):
    """
    Vektor bobot w(x0) untuk local polynomial derajat p.

    w(x0)' = e_1' (Z' W Z)^{-1} Z' W

    dengan Z = [1, (X-x0), (X-x0)^2, ...] dan W = diag(K((X-x0)/h)).
    p=0 -> Nadaraya-Watson;  p=1 -> Local Linear.

    Mengembalikan array (n,). Bobot ini TIDAK bergantung pada Y,
    hanya pada (X, x0, h, p, K) -- inilah yang membuat bias deterministik.
    """
    K = KERNELS[kernel]
    u = (X - x0) / h
    Wd = K(u)

    if np.sum(Wd > 0) < p + 1:          # bandwidth terlalu kecil di titik ini
        return np.full_like(X, np.nan, dtype=float)

    Z = np.vander(X - x0, N=p + 1, increasing=True)   # [1, (X-x0), (X-x0)^2, ...]
    ZtW = Z.T * Wd                                     # (p+1, n)
    A = ZtW @ Z                                        # (p+1, p+1)

    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.full_like(X, np.nan, dtype=float)

    return (Ainv @ ZtW)[0]                             # baris e_1' -> (n,)


def weight_matrix(x_eval, X, h, p, kernel="epanechnikov"):
    """Matriks bobot L, baris ke-j = w(x_eval[j])'. Shape (len(x_eval), n)."""
    return np.vstack([lp_weights(x0, X, h, p, kernel) for x0 in x_eval])


# ---------- Bias eksak (deterministik) ----------
def exact_bias(x_eval, X, h, p, m_func, kernel="epanechnikov"):
    """
    Bias(x) = sum_t w_t(x) m(x_t) - m(x).

    Tidak ada Monte Carlo di sini: pada desain tetap, X dan w(x) tidak acak,
    sehingga E[m_hat(x)] = sum_t w_t(x) m(x_t) dihitung persis.
    """
    L = weight_matrix(x_eval, X, h, p, kernel)
    return L @ m_func(X) - m_func(x_eval)


# ---------- Varians ----------
def exact_variance(x_eval, X, h, p, sigma2, kernel="epanechnikov", Sigma=None):
    """
    Var(m_hat(x)) = w(x)' Cov(eps) w(x).

    Galat i.i.d.  : Cov = sigma2 * I      -> Var = sigma2 * ||w(x)||^2
    Galat AR(1)   : berikan Sigma (n,n) matriks korelasi; Var = sigma2 w' Sigma w
    """
    L = weight_matrix(x_eval, X, h, p, kernel)
    if Sigma is None:
        return sigma2 * np.sum(L**2, axis=1)
    return sigma2 * np.einsum("ij,jk,ik->i", L, Sigma, L)


def ar1_corr(n, rho):
    """Matriks korelasi AR(1): Sigma[i,j] = rho^|i-j|."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# ---------- Estimasi pada data ----------
def fit(x_eval, X, Y, h, p, kernel="epanechnikov"):
    return weight_matrix(x_eval, X, h, p, kernel) @ Y
