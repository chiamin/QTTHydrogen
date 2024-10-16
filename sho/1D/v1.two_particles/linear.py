import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut

def make_t_matrix (n, rescale, power=1):
    return np.array([[0.,0.],[0.,(rescale*2**n)**power]])

def make_x_tensor (n, rescale):
    T = np.zeros((2,2,2,2))     # k1, ipr, i, k2
    T[0,:,:,0] = ut.I
    T[1,:,:,1] = ut.I
    T[1,:,:,0] = make_t_matrix (n, rescale)
    return T

def make_x_LR (shift):
    L = np.array([shift,1.])
    R = np.array([1.,0.])
    return L, R

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

# f(x) = x - a
def make_x_optt (N, shift, rescale):
    tt = [make_x_tensor (n, rescale) for n in range(N)]
    L = np.array([shift,1.])
    R = np.array([1.,0.])
    tt[0] = ncon ([L,tt[0]], ((1,), (1,-1,-2,-3)))
    tt[-1] = ncon ([R,tt[-1]], ((1,), (-1,-2,-3,1)))
    return tt

if __name__ == '__main__':
    N = 4
    shift = 0
    rescale = 0.01
    x_optt = make_x_optt_symm (N, shift, rescale)

    inds = []
    xs,fs = [],[]
    for i4 in [0,1]:
        for i3 in [0,1]:
            for i2 in [0,1]:
                for i1 in [0,1]:
                    inds = [i1,i2,i3,i4]
                    x = ut.inds_to_x (inds, rescale)
                    f = ut.get_ele_op (x_optt, inds)
                    xs.append(x)
                    fs.append(f)
    plt.plot (xs, fs)
    plt.show()
