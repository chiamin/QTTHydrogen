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
from test import load_mps
import npmps
import tci

if __name__ == '__main__':
    '''e = 1.602176634e-19
    hbar = 6.62607015e-34*0.5/np.pi
    me = 9.1093837015e-31
    prefact = me*e**4/(2*hbar**2)
    print(prefact)
    exit()'''


    N = 6
    rescale = 0.1
    cutoff = 0.1

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)



    # Generate the MPS for the potential
    factor = 0.02


    # Kinetic energy
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N)
    Hk, Lk, Rk = hamilt.get_H_3D (Hk1, Lk1, Rk1)
    assert len(Hk) == 3*N

    # Potential energy
    factor = 0.02
    os.system('python3 tci.py '+str(3*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --3D_one_over_r')
    V_MPS = load_mps('fit.mps.npy')
    #V_MPS = tci.tci_one_over_r_2D (2*N, rescale, cutoff, factor, shift)
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    '''bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)
    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, ZV, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()'''

    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    assert len(H) == 3*N

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Create MPS


    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40
    cutoff = 1e-12

    # Run dmrg
    maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40
    psi0 = npmps.random_MPS (3*N, 2, 2)
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (psi0, H, L, R, maxdims, cutoff)
    np.savetxt('terr0.dat',(maxdims,terrs0,ens0))

    #maxdims = maxdims + [64]*80
    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi1 = npmps.random_MPS (3*N, 2, 2)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[20])
    np.savetxt('terr1.dat',(maxdims,terrs1,ens1))

    #maxdims = maxdims + [64]*256
    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi2 = npmps.random_MPS (3*N, 2, 2)
    psi2 = mpsut.npmps_to_uniten (psi2)
    psi2,ens2,terrs2 = dmrg.dmrg (psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    np.savetxt('terr2.dat',(maxdims,terrs2,ens2))

    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi3 = npmps.random_MPS (3*N, 2, 2)
    psi3 = mpsut.npmps_to_uniten (psi3)
    psi3,ens3,terrs3 = dmrg.dmrg (psi3, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2], weights=[20,20,20])
    np.savetxt('terr3.dat',(maxdims,terrs3,ens3))

    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi4 = npmps.random_MPS (3*N, 2, 2)
    psi4 = mpsut.npmps_to_uniten (psi4)
    psi4,ens4,terrs4 = dmrg.dmrg (psi4, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3], weights=[20,20,20,20])
    np.savetxt('terr4.dat',(maxdims,terrs4,ens4))

    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi5 = npmps.random_MPS (3*N, 2, 2)
    psi5 = mpsut.npmps_to_uniten (psi5)
    psi5,ens5,terrs5 = dmrg.dmrg (psi5, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3,psi4], weights=[20,20,20,20,20])
    np.savetxt('terr5.dat',(maxdims,terrs5,ens5))

    print(ens0[-1], ens1[-1], ens2[-1], ens3[-1], ens4[-1], ens5[-1])
    exit()

    phi0 = mpsut.to_npMPS (psi0)
    phi1 = mpsut.to_npMPS (psi1)
    phi2 = mpsut.to_npMPS (psi2)
    phi3 = mpsut.to_npMPS (psi3)


    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)

    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    Z0 = ptut.get_2D_mesh_eles_mps (phi0, bxs, bys)
    Z1 = ptut.get_2D_mesh_eles_mps (phi1, bxs, bys)
    Z2 = ptut.get_2D_mesh_eles_mps (phi2, bxs, bys)
    Z3 = ptut.get_2D_mesh_eles_mps (phi3, bxs, bys)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, np.square(Z0), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, np.square(Z1), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, np.square(Z2), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, np.square(Z3), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

