import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon

def make_LR_sin (shift=0.):
    L = np.array([np.sin(shift),np.cos(shift)])
    R = np.array([1.,0.])
    return L, R

def make_LR_cos (shift=0.):
    L = np.array([np.cos(shift),-np.sin(shift)])
    R = np.array([1.,0.])
    return L, R

def make_tensorA (n, rescale=1.):
    A = np.zeros ((2,2,2)) # (k1,i,k2)
    x = rescale * 2**n
    # k1 == 0, k2 == 0
    A[0,0,0] = 1
    A[0,1,0] = np.cos(x)
    # k1 == 1, k2 == 0
    A[1,0,0] = 0
    A[1,1,0] = np.sin(x)
    # k1 == 0, k2 == 1
    A[0,0,1] = 0
    A[0,1,1] = -np.sin(x)
    # k1 == 1, k2 == 1
    A[1,0,1] = 1
    A[1,1,1] = np.cos(x)
    return A

def sin_mps (N, x1, x2):
    Ndx = 2**N-1
    rescale = (x2-x1)/Ndx
    shift = x1

    shift_i = shift/N
    mps = []
    for i in range(N):
        mps.append (make_tensorA(i, rescale))
    L, R = make_LR_sin(shift)
    mps[0] = ncon([L,mps[0]], [(1,),(1,-1,-2)])
    mps[0] = mps[0].reshape((1,*mps[0].shape))
    mps[-1] = ncon([R,mps[-1]], [(1,),(-1,-2,1)])
    mps[-1] = mps[-1].reshape((*mps[-1].shape,1))
    return mps

def cos_mps (N, x1, x2):
    Ndx = 2**N-1
    rescale = (x2-x1)/Ndx
    shift = x1

    shift_i = shift/N
    mps = []
    for i in range(N):
        mps.append (make_tensorA(i, rescale))
    L, R = make_LR_cos(shift)
    mps[0] = ncon([L,mps[0]], [(1,),(1,-1,-2)])
    mps[0] = mps[0].reshape((1,*mps[0].shape))
    mps[-1] = ncon([R,mps[-1]], [(1,),(-1,-2,1)])
    mps[-1] = mps[-1].reshape((*mps[-1].shape,1))
    return mps

