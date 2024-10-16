import differential as df
import npmps
import numpy as np

def add_spin_to_H (H, L, R):
    H_I, L_I, R_I = npmps.identity_MPO (1, 2)
    Hspin, Lspin, Rspin = npmps.product_2MPO (H, L, R, H_I, L_I, R_I)
    return Hspin, Lspin, Rspin

def get_H_2D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_1D, L_1D, R_1D, H_I, L_I, R_I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R
