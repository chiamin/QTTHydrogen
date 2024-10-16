import numpy as np
from ncon import ncon

I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])

def make_tensorA_df2 ():
    A = np.zeros ((3,2,2,3)) # (k1,ipr,i,k2)
    A[0,:,:,0] = I
    A[1,:,:,0] = sp
    A[2,:,:,0] = sm
    A[1,:,:,1] = sm
    A[2,:,:,2] = sp
    return A

def make_LR_df2 ():
    L = np.array([-2.,1.,1.])
    R = np.array([1.,1.,1.])
    return L, R

def diff2_mpo (N, x1, x2):
    dx = (x2-x1)/(2**N-1)
    mpo = []
    for i in range(N):
        mpo.append (make_tensorA_df2())
    L, R = make_LR_df2()
    L *= 1./dx**2
    return mpo, L, R

def make_tensorA_df ():
    A = np.zeros ((2,2,2,2)) # (k1,ipr,i,k2)
    A[0,:,:,0] = I
    A[1,:,:,0] = sp
    A[1,:,:,1] = sm
    return A

def make_LR_df ():
    L = np.array([-1.,1.])
    R = np.array([1.,1.])
    return L, R

def diff_mpo (N, x1, x2):
    dx = (x2-x1)/(2**N-1)
    mpo = []
    for i in range(N):
        mpo.append (make_tensorA_df())
    L, R = make_LR_df()
    L *= 1./dx
    return mpo, L, R
