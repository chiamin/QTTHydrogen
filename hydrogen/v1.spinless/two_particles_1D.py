import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys, os
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
from tci import load_mps
import npmps
import plotsetting as ps

if __name__ == '__main__':
    N = 10
    x1,x2 = -2,2
    cutoff = 0.0002
    factor = 0.0008

    xmax = 2**N-1
    rescale = (x2-x1)/xmax
    shift = x1


    #xmax = rescale * (2**N-1)
    #shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)


    Hk, Lk, Rk = hamilt.H_kinetic(N)

    # Generate the MPS for the potential
    os.system('python3 tci.py '+str(N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --1D_one_over_r')

    # Load the potential MPS
    V_MPS = load_mps('fit.mps.npy')
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    print(len(Hk),len(HV))
    H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)

    # Create a two-particle Hamiltonian
    H, L, R = twut.H_two_particles (H1, L1, R1)

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps (N)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40
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

    overlap = mpsut.inner (phi1, phi2)

    print('E =',en)
    print('occ =',occ1, occ2)
    print('inner product of the orbitals =', overlap)

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    xs = xs*rescale + shift

    # The potential
    Vx = [ptut.get_ele_mps (V_MPS, bx) for bx in bxs]

    # First particle
    npphi1 = mpsut.to_npMPS (phi1)
    ys1 = [ptut.get_ele_mps (npphi1, bx) for bx in bxs]

    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)
    ys2 = [ptut.get_ele_mps (npphi2, bx) for bx in bxs]

    fig, ax = plt.subplots()
    ax.plot (xs, Vx)
    ax.plot (xs, ys1, marker='.')
    ax.plot (xs, ys2, marker='.')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\phi(x)$')
    ps.set(ax)
    fig.savefig('phi.pdf')
    plt.show()

