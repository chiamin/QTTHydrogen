import polynomial as poly
import differential as diff
import npmps
import numpy as np
import matplotlib.pyplot as plt

def make_H (N, x1, x2, hbar=1, m=1, omega=1):
    Hk, Lk, Rk = diff.diff2_mpo (N, x1, x2)
    HV, LV, RV = poly.make_xsqr_mpo (N, x1, x2)
    coef_k = -0.5*hbar**2/m
    coef_V = 0.5*m*omega**2
    H, L, R = npmps.sum_2MPO (Hk, coef_k*Lk, Rk, HV, coef_V*LV, RV)
    H, L, R = npmps.compress_mpo (H, L, R, cutoff=1e-12)
    return H, L, R

def plot_GS_exact (x1, x2, ax, hbar=1, m=1, omega=1, **args):
    def gs (x):
        return (m*omega/(np.pi*hbar))**0.25 * np.exp(-m*omega*x*x*0.5/hbar)

    xs = np.linspace(x1,x2,200)
    ys = [gs(i) for i in xs]
    ax.plot(xs,ys, **args)

def plot_1ES_exact (x1, x2, ax, hbar=1, m=1, omega=1, **args):
    def es (x):
        return (m*omega/(np.pi*hbar*4))**0.25 * np.exp(-m*omega*x*x*0.5/hbar) * (2*(m*omega/hbar)**0.5*x)

    xs = np.linspace(x1,x2,200)
    ys = [es(i) for i in xs]
    ax.plot(xs,ys, **args)

def exact_energy (n, hbar=1, omega=1):
    return (n+0.5)*hbar*omega
