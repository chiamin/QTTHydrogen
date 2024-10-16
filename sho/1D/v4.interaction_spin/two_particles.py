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

def project_spin (mps, spin):
    mps = copy.copy(mps)
    mps[-1] = mps[-1][:,spin,:]
    mps[-2] = ncon([mps[-2],mps[-1]], ((-1,-2,1),(1,-3)))
    return mps[:-1]

def project_tot_spin (mps):
    Ps = np.zeros((4,4))
    Pt = np.zeros((4,4))
    Ps[3,3] = 1.
    Pt[0,0] = Pt[1,1] = Pt[2,2] = 1.
    mps_s = copy.copy(mps)
    mps_t = copy.copy(mps)
    mps_s = npmps.apply_op_MPS (mps_s, Ps, -1)
    mps_t = npmps.apply_op_MPS (mps_t, Pt, -1)
    return mps_s, mps_t

if __name__ == '__main__':
    N = 6
    rescale = 0.05

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    # One-body H
    Hk, Lk, Rk = hamilt.H_kinetic(N)

    '''Hmat = npmps.MPO_to_matrix (Hk, Lk, Rk)
    Hmat = Hmat.round(2)
    print(Hmat)'''

    Hsho, Lsho, Rsho = sho.V_SHO_MPO (N, rescale)

    '''Hmat = npmps.MPO_to_matrix (Hsho, Lsho, Rsho)
    Hmat = Hmat.round(20)
    print(Hmat)'''



    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)

    # First particle
    y = [ptut.get_ele_mpo (Hsho, Lsho, Rsho, bx) for bx in bxs]
    fig, ax = plt.subplots()
    ax.plot (xs, y, marker='.')
    plt.show()

    H1, L1, R1 = npmps.sum_2MPO (Hk, -Lk, Rk, Hsho, Lsho, Rsho)


    #Hmat = npmps.MPO_to_matrix (H1, L1, R1)
    #Hmat = Hmat.round(2)
    #print(Hmat)
    #exit()


    H, L, R = twut.H_two_particles (H1, L1, R1)

    # Interaction
    os.system('python3 tci.py '+str(2*N)+' '+str(rescale))
    inter_MPS = load_mps('fit.mps.npy')
    Hint, Lint, Rint = npmps.mps_func_to_mpo (inter_MPS)
    Hint = twut.H_merge_two_particle (Hint)
    Hint[0] *= 6

    H, L, R = npmps.sum_2MPO (H, L, R, Hint, Lint, Rint)

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    H, L, R = hamilt.add_spin_to_H (H, L, R, dim=4)

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps_spin (N)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*20 + [8]*40 + [16]*40 + [32]*40
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)

    # Run dmrg
    psi, ens = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4, verbose=False)

    # ------- Compute the single-particle correlation function -----------
    nppsi = ut.mps_to_nparray (psi)

    # Project to spin singlet or triplet sector
    nppsi_s, nppsi_t = project_tot_spin (nppsi)
    c_s = npmps.inner_MPS (nppsi_s, nppsi_s)
    c_t = npmps.inner_MPS (nppsi_t, nppsi_t)

    nppsi = ut.applyLocal_mps (nppsi, U)

    corr = twut.corr_matrix (nppsi)
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
    #print(npphi1[-1].shape)
    #print(npphi2[-1].shape)
    #print(npphi1[-1], npphi2[-1])
    npphi1_up = project_spin (npphi1, 0)
    npphi1_dn = project_spin (npphi1, 1)
    npphi2_up = project_spin (npphi2, 0)
    npphi2_dn = project_spin (npphi2, 1)

    assert abs(np.linalg.norm(npphi1[-1]))-1. < 1e-12 and abs(np.linalg.norm(npphi2[-1]))-1. < 1e-12
    assert abs(mpsut.inner(phi1,phi1))-1. < 1e-12 and abs(mpsut.inner(phi2,phi2))-1. < 1e-12
    overlap = mpsut.inner (phi1, phi2)
    print('E =',ens[-1])
    print('occ =',occ1, occ2)
    print('inner product =', overlap)

    print('spin contributions',c_s,c_t)
    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)

    # First particle
    ys1_up = [ptut.get_ele_mps (npphi1_up, bx) for bx in bxs]
    ys1_dn = [ptut.get_ele_mps (npphi1_dn, bx) for bx in bxs]

    # Second particle
    ys2_up = [ptut.get_ele_mps (npphi2_up, bx) for bx in bxs]
    ys2_dn = [ptut.get_ele_mps (npphi2_dn, bx) for bx in bxs]

    fig, ax = plt.subplots()
    ax.plot (xs, ys1_up, marker='.', label='1 up')
    ax.plot (xs, ys1_dn, marker='x', label='1 dn')
    ax.plot (xs, ys2_up, marker='.', label='2 up')
    ax.plot (xs, ys2_dn, marker='x', label='2 dn')
    plt.legend()
    plt.show()

