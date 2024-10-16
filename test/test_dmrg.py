import sys
sys.path.append('../tools')
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import polynomial as poly
import differential as diff
import npmps
import unitenMPS as mpsut
import plot_utility as ptut
import hamilt.hamilt_sho as sho

def straight_dmrg (ax):
    N = 14
    x1,x2 = -5,5
    shift = 1

    H = sho.make_H (N, x1, x2, shift=shift)
    psi = npmps.random_MPS (N,2,2)

    # Define the bond dimensions for the sweeps
    nsweep = 20
    maxdims = [2]*nsweep + [4]*nsweep + [8]*nsweep + [16]*nsweep + [32]*nsweep
    cutoff = 1e-12

    psi0, ens0, terrs0 = dmrg.dmrg (2, psi0, H, maxdims, cutoff)
    for i,j in zip(ens0,terrs0):
        print(i,j)

    # Plot
    mps = mpsut.mps_to_nparray (psi0)
    mps = npmps.normalize_by_integral (mps, x1, x2)
    ptut.plot_1D (mps, x1, x2, ax=ax)

    en0 = sho.exact_energy(0)
    en_errs = [en-en0 for en in ens0]
    plt.figure()
    plt.plot (range(len(ens0)), en_errs, marker='o')
    plt.yscale('log')

def dmrg_increase_site (ax):
    Nbeg = 5
    N = 14
    x1,x2 = -5,5

    Ndx = 2**Nbeg-1
    dx = (x2-x1)/Ndx


    # --- First run ---
    H, L, R = sho.make_H (Nbeg, x1, x2)      # harmonic oscillator
    psi = npmps.random_MPS (Nbeg,2,2)

    # Define the bond dimensions for the sweeps
    nsweep = 4
    maxdims = [2]*nsweep + [4]*nsweep# + [8]*nsweep + [16]*nsweep + [32]*nsweep
    cutoff = 1e-12

    # Run dmrg
    psi = mpsut.npmps_to_uniten (psi)
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)
    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)

    # Plot
    mps = mpsut.mps_to_nparray (psi)
    mps[0] /= dx**0.5
    ptut.plot_1D (mps, x1, x2, ax=ax, marker='.',label='DMRG')
    # -----------------



    # --- Increase site ---

    # Add one site
    A = np.array([1.,1.]).reshape(1,2,1)
    A = mpsut.np_to_MPS_tensor(A)
    for Ni in range(Nbeg+1,N+1):
        psi = [A] + psi
        dx *= 0.5

    # Make H
    H, L, R = sho.make_H (N, x1, x2)
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Run dmrg
    nsweep = 4
    maxdims = [2]*nsweep + [4]*nsweep + [8]*nsweep + [16]*nsweep + [32]*nsweep
    cutoff = 1e-12

    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)
    print(ens)

    # Plot
    mps = mpsut.mps_to_nparray (psi)
    mps[0] /= dx**0.5
    ptut.plot_1D (mps, x1, x2, ax=ax,label='DMRG')

    en0 = sho.exact_energy(0)
    en_errs = [en-en0 for en in ens]
    plt.figure()
    plt.plot (range(len(ens)), en_errs, marker='o')
    plt.yscale('log')

if __name__ == '__main__':
    fig, ax = plt.subplots()
    sho.plot_GS_exact(-4,4,ax,ls='--',label='Exact')
    straight_dmrg(ax)
    #dmrg_increase_site(ax)
    plt.legend()
    plt.show()
