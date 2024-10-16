import numpy as np
import copy
import npmps

def grow_site_basic (psi, normalize=True):
    A = np.array([1.,1.]).reshape(1,2,1)
    if normalize:
        A /= np.linalg.norm(A)
    return [A] + psi

def grow_site_1D (psi):
    psi = copy.copy(psi)
    ds = psi[0].shape

    A1 = np.zeros((2,ds[1],ds[2]))
    A1[0,0,:] = psi[0][0,0,:]
    A1[1,1,:] = psi[0][0,1,:]
    psi[0] = A1

    A0 = np.zeros((1,2,2))
    A0[0,0,:] = [1,1]
    A0[0,1,:] = [1,1]
    mps = [A0] + psi
    return mps

def grow_site_2D (psi):
    assert len(psi) % 2 == 0
    psi = copy.copy(psi)
    ds = psi[0].shape

    # For x
    A1 = np.zeros((2,ds[1],ds[2]))
    A1[0,0,:] = psi[0][0,0,:]
    A1[1,1,:] = psi[0][0,1,:]
    psi[0] = A1

    A0 = np.zeros((1,2,2))
    A0[0,0,:] = [1,1]
    A0[0,1,:] = [1,1]

    # For y
    ind = len(psi) // 2
    bdim = psi[ind].shape[0]    # bond dimension
    pdim = psi[0].shape[1]      # physical dimension
    A2 = np.zeros((bdim, pdim, bdim))
    for i in range(pdim):
        A2[:,i,:] = np.ones ((bdim,bdim))#bdim)

    mps = [A0] + psi[:ind] + [A2] + psi[ind:]

    npmps.check_MPS_links(mps)
    return mps

def MPS_tensor_to_MPO_tensor (A):
    assert A.ndim == 3
    T = np.zeros((A.shape[0],A.shape[1],A.shape[1],A.shape[2]), dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[2]):
            ele = A[i,:,j]
            T[i,:,:,j] = np.diag(ele)
    return T

def MPS_to_MPO (mps):
    npmps.check_MPS_links (mps)

    mpo = []
    for A in mps:
        T = MPS_tensor_to_MPO_tensor (A)
        mpo.append(T)
    return mpo

def normalize_MPS_by_integral (mps, x1, x2, Dim):
    mps = copy.copy(mps)
    c = npmps.inner_MPS (mps, mps)
    mps[0] = mps[0] / c**0.5

    N = len(mps)//Dim
    Ndx = 2**N-1
    dx = (x2-x1)/Ndx

    for d in range(Dim):
        i = d*N
        mps[i] = mps[i] / dx**0.5
    return mps

def sum_elements (mps):
    A = np.array([1,1]).reshape((1,2,1))
    mps2 = [A for i in range(len(mps))]
    return npmps.inner_MPS (mps, mps2)
