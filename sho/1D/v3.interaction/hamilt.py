import differential as df
import npmps
import numpy as np

def get_H (V_MPS):
    V_MPO, VL, VR = npmps.mps_func_to_mpo (V_MPS)

    H = []
    for n in range(len(V_MPS)):
        ddx2_tensor = df.make_tensorA()
        hi = npmps.sum_mpo_tensor (ddx2_tensor, V_MPO[n])
        H.append(hi)

    L_ddx2, R_ddx2 = df.make_LR()
    L = np.concatenate ((L_ddx2, VL))
    R = np.concatenate ((R_ddx2, VR))
    return H, L, R

def H_kinetic (N):
    H = []
    for n in range(N):
        ddx2_tensor = df.make_tensorA()
        H.append(ddx2_tensor)
    L, R = df.make_LR()
    return H, L, R

def get_H_2D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_1D, L_1D, R_1D, H_I, L_I, R_I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R
