import qtt_utility as ut
from ncon import ncon
import numpy as np
import matplotlib.pyplot as plt

def make_t_matrix (n, rescale, power=1):
    return np.array([[0.,0.],[0.,(rescale*2**n)**power]])

def make_x_tensor (n, rescale):
    T = np.zeros((2,2,2,2))     # k1, ipr, i, k2
    T[0,:,:,0] = ut.I
    T[1,:,:,1] = ut.I
    T[1,:,:,0] = make_t_matrix (n, rescale)
    return T

def make_x_MPS_tensor (n, rescale):
    T = np.zeros((2,2,2))     # k1, i, k2
    T[0,:,0] = [1.,1.]
    T[1,:,1] = [1.,1.]
    T[1,:,0] = [0.,rescale*2**n]
    return T

def make_x_LR (shift):
    L = np.array([shift,1.])
    R = np.array([1.,0.])
    return L, R

def make_x_mps (N, shift=0., rescale=1.):
    mps = [make_x_MPS_tensor (n, rescale) for n in range(N)]
    L, R = make_x_LR (shift)
    mps[0] = ncon ([L,mps[0]], ((1,), (1,-1,-2))).reshape((1,2,2))
    mps[-1] = ncon ([R,mps[-1]], ((1,), (-1,-2,1))).reshape((2,2,1))
    return mps

def make_x_mpo (N, shift=0., rescale=1.):
    mpo = [make_x_tensor (n, rescale) for n in range(N)]
    L, R = make_x_LR (shift)
    return mpo, L, R

def contract_L (mpo, L):
    mpo[0] = ncon ([L,mpo[0]], ((1,), (1,-1,-2,-3)))
    return mpo

def contract_R (mpo, R):
    mpo[-1] = ncon ([R,mpo[-1]], ((1,), (-1,-2,-3,1)))
    return mpo

def contract_LR (mpo, L, R):
    mpo = contract_L (mpo, L)
    mpo = contract_R (mpo, R)
    return mpo

if __name__ == '__main__':
    N = 4
    shift = 0
    rescale = 0.01
    x_mps = make_x_mps (N, shift, rescale)

    inds = []
    xs,fs = [],[]
    for i4 in [0,1]:
        for i3 in [0,1]:
            for i2 in [0,1]:
                for i1 in [0,1]:
                    inds = [i1,i2,i3,i4]
                    x = ut.inds_to_x (inds, rescale)
                    f = ut.get_ele_mps (x_mps, inds)
                    xs.append(x)
                    fs.append(f)
    plt.plot (xs, fs)
    plt.show()
