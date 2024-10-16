import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys
sys.path.append('/home/chiamin/mypy/')
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import qn_utility as qn
import MPS_utility as mpsut
import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
from test import load_mps
import npmps
import plotsetting as ps

if __name__ == '__main__':
    N = 6
    rescale = 0.05

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    # One-body H
    Hk, Lk, Rk = hamilt.H_kinetic(N)
    Hsho, Lsho, Rsho = sho.V_SHO_MPO (N, rescale)

    H1, L1, R1 = npmps.sum_2MPO (Hk, -Lk, Rk, Hsho, Lsho, Rsho)
    H, L, R = twut.H_two_particles (H1, L1, R1)

    # Interaction
    inter_MPS = load_mps('fit.mps.npy')
    Hint, Lint, Rint = npmps.mps_func_to_mpo (inter_MPS)
    Hint = twut.H_merge_two_particle (Hint)
    Hint[0] *= 4.

    H, L, R = npmps.sum_2MPO (H, L, R, Hint, Lint, Rint)

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps (N)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40 + [64]*40
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)

    # Run dmrg
    psi, en = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)

    # ------- Compute the single-particle correlation function -----------
    psi = ut.mps_to_nparray (psi)
    psi = ut.applyLocal_mps (psi, U)

    corr = twut.corr_matrix (psi)
    Lcorr = np.array([1.])
    Rcorr = np.array([1.])

    # Target the largest occupations
    corr[0] *= -1.
    corr, Lcorr, Rcorr = ut.compress_mpo (corr, Lcorr, Rcorr, cutoff=1e-12)

    maxdims = [2]*10 + [4]*10 + [8]*20 + [16]*40

    corr, Lcorr, Rcorr = ut.mpo_to_uniten (corr, Lcorr, Rcorr)
    phi1 = ut.generate_random_MPS_nparray (N, d=2, D=2)
    phi1 = ut.mps_to_uniten (phi1)

    phi1, occ1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff)

    phi2 = ut.generate_random_MPS_nparray (N, d=2, D=2)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2 = dmrg.dmrg (phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[20])

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)

    # First particle
    npphi1 = mpsut.to_npMPS (phi1)
    ys1 = [ptut.get_ele_mps (npphi1, bx) for bx in bxs]

    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)
    ys2 = [ptut.get_ele_mps (npphi2, bx) for bx in bxs]

    fig, ax = plt.subplots()
    ax.plot (xs, ys1, marker='.')
    ax.plot (xs, ys2, marker='.')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\phi(x)$')
    ps.set(ax)
    fig.savefig('phi.pdf')
    plt.show()

