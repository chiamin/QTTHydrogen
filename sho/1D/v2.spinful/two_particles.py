import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import qn_utility as qn
import MPS_utility as mpsut
import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
import npmps

if __name__ == '__main__':
    N = 8
    rescale = 0.01

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    H1, L1, R1 = sho.get_H_SHO (N, rescale)
    print(len(H1))
    H1, L1, R1 = hamilt.add_spin_to_H (H1, L1, R1)
    print(len(H1))

    H, L, R = twut.H_two_particles (H1, L1, R1)
    print(len(H))

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps_spin (N)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*20 + [16]*80# + [32]*30 + [64]*20
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
    phi1 = ut.generate_random_MPS_nparray (N+1, d=2, D=2)
    phi1 = ut.mps_to_uniten (phi1)

    phi1, occ1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff)

    phi2 = ut.generate_random_MPS_nparray (N+1, d=2, D=2)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2 = dmrg.dmrg (phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[20])

    npphi1 = mpsut.to_npMPS (phi1)
    npphi2 = mpsut.to_npMPS (phi2)
    assert npphi1[-1].shape[0] == 1
    assert npphi2[-1].shape[0] == 1
    npphi1sp = npphi1[:-1]
    npphi2sp = npphi2[:-1]

    assert abs(np.linalg.norm(npphi1[-1]))-1. < 1e-12 and abs(np.linalg.norm(npphi2[-1]))-1. < 1e-12
    assert abs(mpsut.inner(phi1,phi1))-1. < 1e-12 and abs(mpsut.inner(phi2,phi2))-1. < 1e-12
    overlap = mpsut.inner (phi1, phi2)
    overlap_space = mpsut.inner (phi1[:-1], phi2[:-1])
    print('E =',en)
    print('occ =',occ1, occ2)
    print('inner product =', overlap, overlap_space)
    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)

    # First particle
    ys1 = [ptut.get_ele_mps (npphi1sp, bx) for bx in bxs]

    # Second particle
    ys2 = [ptut.get_ele_mps (npphi2sp, bx) for bx in bxs]

    e1 = npmps.inner_MPO (npphi1, npphi1, H1, L1, R1)
    e2 = npmps.inner_MPO (npphi2, npphi2, H1, L1, R1)
    print(e1,e2,e1+e2)

    fig, ax = plt.subplots()
    ax.plot (xs, ys1, marker='.')
    ax.plot (xs, ys2, marker='.')
    plt.show()

