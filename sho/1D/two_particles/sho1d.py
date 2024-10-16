import copy, sys
sys.path.append('/home/chiamin/mypy/')
sys.path.append('/home/chiamin/project/2023/qtt/code/new/tools/')
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
import dmrg as dmrg
import matplotlib.pyplot as plt
#import polynomial as poly
#import differential as diff
import npmps
import unitenMPS as uniten
import plot_utility as ptut
import hamilt.hamilt_sho as sho
import two_particles as tpar

def make_H (N, x1, x2, dim=1):
    H1, L1, R1 = sho.make_H (N, x1, x2)
    if dim == 2:
        H1, L1, R1 = npmps.get_H_2D (H1, L1, R1)
    elif dim == 3:
        H1, L1, R1 = npmps.get_H_3D (H1, L1, R1)

    H, L, R = npmps.H1_to_two_particles (H1, L1, R1)   # from one-particle to two-particle hamiltonian
    H, L, R = npmps.compress_mpo (H, L, R, cutoff=1e-12)

    # Rotate H to the parity basis
    U = tpar.get_swap_U()
    H = npmps.applyLocalRot_mpo (H, U)
    # Set quantum numbers
    H, L, R = tpar.set_mpo_quantum_number (H, L, R)
    return H, L, R

if __name__ == '__main__':
    Nbeg = 5
    N = 14
    x1,x2 = -5,5

    Ndx = 2**Nbeg-1
    dx = (x2-x1)/Ndx

    fig, ax = plt.subplots()

    # ------------ Pre-run DMRG -------------
    H, L, R = make_H (Nbeg, x1, x2)

    maxdims = [2]*4 + [4]*4 + [8]*4
    cutoff = 1e-12

    psi = tpar.make_product_mps (Nbeg)
    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)

    # Plot natural orbitals
    maxdims = [2]*10 + [4]*10 + [8]*20 + [16]*20
    U = tpar.get_swap_U()
    phi1, phi2 = tpar.one_particle_denmat (psi, U, maxdims, cutoff=1e-10)

    phi1 = npmps.normalize_by_integral (phi1, x1, x2)
    phi2 = npmps.normalize_by_integral (phi2, x1, x2)
    ptut.plot_1D (phi1, x1, x2, ax=ax, marker='.')
    ptut.plot_1D (phi2, x1, x2, ax=ax, marker='.')

    # ------------- Increase site DMRG -------------
    # Add sites
    psi = tpar.add_mps_sites (psi, N-Nbeg)

    # Make H
    H, L, R = make_H (N, x1, x2)

    # Run dmrg
    nsweep = 20
    maxdims = [8]*nsweep + [16]*nsweep# + [32]*nsweep
    cutoff = 1e-12

    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=10)
    fig2, ax2 = plt.subplots()
    en_exact = sho.exact_energy(0) + sho.exact_energy(1)
    en_errs = [i-en_exact for i in ens]
    ax2.plot(range(len(ens)),en_errs,marker='o')
    ax2.set_yscale('log')


    # ------- Compute the single-particle correlation function -----------
    maxdims = [2]*10 + [4]*10 + [8]*20 + [16]*40
    U = tpar.get_swap_U()
    phi1, phi2 = tpar.one_particle_denmat (psi, U, maxdims, cutoff=1e-10)

    # --------------- Plot ---------------------
    phi1 = npmps.normalize_by_integral (phi1, x1, x2)
    phi2 = npmps.normalize_by_integral (phi2, x1, x2)
    ptut.plot_1D (phi1, x1, x2, ax=ax, marker='x')
    ptut.plot_1D (phi2, x1, x2, ax=ax, marker='x')

    sho.plot_GS_exact (x1, x2, ax)
    sho.plot_1ES_exact (x1, x2, ax)
    plt.show()
