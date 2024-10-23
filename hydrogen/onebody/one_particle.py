import copy, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import dmrg as dmrg
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import numpy as np
import qn_utility as qn
import MPS_utility as mpsut
import npmps
import plot_utility as ptut
from matplotlib import cm

def get_H_SHO (N, rescale):
    xmax = rescale * (2**N-1)
    shift = -xmax/2
    #print('xmax =',xmax)
    #print('xshift =',shift)

    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        x2_tensor = ut.prod_mpo_tensor (x_tensor, x_tensor)
        ddx2_tensor = df.make_tensorA()
        hi = ut.sum_mpo_tensor (x2_tensor, ddx2_tensor)
        H.append(hi)

    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    L_ddx2, R_ddx2 = df.make_LR()
    L = np.concatenate ((L_x2, -L_ddx2))
    R = np.concatenate ((R_x2, R_ddx2))
    return H, L, R

def get_H_SHO_2D (N, rescale):
    H_SHO, L_SHO, R_SHO = get_H_SHO (N, rescale)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_SHO, L_SHO, R_SHO, H_I, L_I, R_I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_SHO, L_SHO, R_SHO)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R

def get_H_xsqrt (N, rescale):
    xmax = rescale * (2**N-1)
    shift = -xmax/2

    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        x2_tensor = ut.prod_mpo_tensor (x_tensor, x_tensor)
        H.append(x2_tensor)

    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    return H, L_x2, R_x2

def get_H_rsqrt_2D (N, rescale):
    H_x2, L_x2, R_x2 = get_H_xsqrt (N, rescale)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_x2, L_x2, R_x2, H_I, L_I, R_I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_x2, L_x2, R_x2)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R

if __name__ == '__main__':
    N = 9
    rescale = 1/2**(N+1)

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    H, L, R = get_H_SHO_2D (N, rescale)
    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)
    npmps.check_MPO_links (H, L, R)
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Get initial state MPS
    psi = npmps.random_MPS (2*N, phydim=2, vdim=2)
    npmps.check_MPS_links (psi)
    psi = mpsut.npmps_to_uniten (psi)

    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*10 + [16]*10 + [32]*30
    cutoff = 1e-12

    # Run dmrg
    psi0, en0 = dmrg.dmrg (psi, H, L, R, maxdims, cutoff)



    # Plot
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)

    nppsi = mpsut.to_npMPS (psi0)
    fs = ptut.get_2D_mesh_eles_mps (nppsi, bxs, bys)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, fs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

