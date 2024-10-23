import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys, os
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import qn_utility as qn
import MPS_utility as mpsut
import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
from tci import load_mps
import npmps

if __name__ == '__main__':
    N = 6
    N_2D = 2*N
    rescale = 0.1
    cutoff = 0.0001
    factor = 0.02

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    # Kinetic energy
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N)
    Hk, Lk, Rk = hamilt.get_H_2D (Hk1, Lk1, Rk1)

    # Potential energy
    factor = 0.02
    os.system('python3 tci.py '+str(2*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --2D_one_over_r')
    V_MPS = load_mps('fit.mps.npy')
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)
    assert len(V_MPS) == 2*N

    H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    assert len(H1) == 2*N

    # Create a two-particle Hamiltonian
    H, L, R = twut.H_two_particles (H1, L1, R1)
    assert len(H) == N_2D


    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps (N_2D)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40# + [32]*30
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
    phi1 = ut.generate_random_MPS_nparray (N_2D, d=2, D=2)
    phi1 = ut.mps_to_uniten (phi1)

    phi1, occ1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff)

    phi2 = ut.generate_random_MPS_nparray (N_2D, d=2, D=2)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2 = dmrg.dmrg (phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[20])

    overlap = mpsut.inner (phi1, phi2)

    print('E =',en)
    print('occ =',occ1, occ2)
    print('inner product of the orbitals =', overlap)

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)

    # First particle
    npphi1 = mpsut.to_npMPS (phi1)
    Z1 = ptut.get_2D_mesh_eles_mps (npphi1, bxs, bys)

    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)
    Z2 = ptut.get_2D_mesh_eles_mps (npphi2, bxs, bys)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

