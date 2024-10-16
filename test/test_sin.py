import sys, copy
sys.path.append('../tools')
import sincos as sc
import plot_utility as ptut
import matplotlib.pyplot as plt
import numpy as np
import differential as dff
import npmps

if __name__ == '__main__':
    N = 6
    x1,x2 = -np.pi,np.pi

    fig, ax = plt.subplots()

    # sin function
    f = sc.sin_mps (N, x1, x2)
    ptut.plot_1D(f, x1, x2, ax=ax, marker='.')

    # first derivative
    d = dff.diff_MPO(N, x1, x2)
    df = npmps.exact_apply_MPO(d, f)
    npmps.check_MPS_links (df)
    ptut.plot_1D(df, x1, x2, ax=ax, marker='x')

    # first derivative
    d = dff.diff_MPO_not_antisymm(N, x1, x2)
    df = npmps.exact_apply_MPO(d, f)
    npmps.check_MPS_links (df)
    ptut.plot_1D(df, x1, x2, ax=ax, marker='*')

    # second derivative
    d2 = dff.diff2_MPO(N, x1, x2)
    d2f = npmps.exact_apply_MPO(d2, f)
    npmps.check_MPS_links (d2f)
    ptut.plot_1D(d2f, x1, x2, ax=ax, marker='+')

    plt.show()
