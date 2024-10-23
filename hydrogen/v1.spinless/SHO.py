import numpy as np
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import npmps

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
