import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys, os
#sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
sys.path.insert(0,'/home/chiamin/cytnx_new/')
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

if __name__ == '__main__':
    N = 10
    rescale = 0.1
    cutoff = 0.1

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)


    Hk, Lk, Rk = hamilt.H_kinetic(N)

    # Generate the MPS for the potential
    factor = 0.02
    os.system('python3 tci.py '+str(N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --1D_one_over_r')

    # Load the potential MPS
    V_MPS = load_mps('fit.mps.npy')
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    print(len(Hk),len(HV))
    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Create MPS


    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*100 + [32]*100
    cutoff = 1e-12

    # Run dmrg
    psi0 = npmps.random_MPS (N, 2, 2)
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (psi0, H, L, R, maxdims, cutoff)
    np.savetxt('terr0.dat',(terrs0,ens0))

    #maxdims = maxdims + [64]*80
    psi1 = npmps.random_MPS (N, 2, 2)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[20])
    np.savetxt('terr1.dat',(terrs1,ens1))

    #maxdims = maxdims + [64]*256
    '''psi2 = npmps.random_MPS (N, 2, 2)
    psi2 = mpsut.npmps_to_uniten (psi2)
    psi2,ens2,terrs2 = dmrg.dmrg (psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    np.savetxt('terr2.dat',(terrs2,ens2))'''

    print(ens0[-1], ens1[-1])#, ens2[-1])


    phi0 = mpsut.to_npMPS (psi0)
    phi1 = mpsut.to_npMPS (psi1)
    #phi2 = mpsut.to_npMPS (psi2)


    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)


    # The potential
    Vx = [ptut.get_ele_mps (V_MPS, bx) for bx in bxs]

    ys0 = [ptut.get_ele_mps (phi0, bx) for bx in bxs]
    ys1 = [ptut.get_ele_mps (phi1, bx) for bx in bxs]
    #ys2 = [ptut.get_ele_mps (phi2, bx) for bx in bxs]

    fig, ax = plt.subplots()
    ax.plot (xs, Vx)
    ax.plot (xs, ys0, marker='.')
    ax.plot (xs, ys1, marker='.')
    #ax.plot (xs, ys2, marker='.')
    plt.show()

