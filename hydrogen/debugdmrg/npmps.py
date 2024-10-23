import numpy as np
from ncon import ncon
import sys
import npUtility as nput
import copy

#======================== MPS #========================
# For each tensor, the order of index is (left, i, right)

def check_MPS_links (mps):
    for i in range(len(mps)):
        assert mps[i].ndim == 3
        if i != 0:
            assert mps[i-1].shape[-1] == mps[i].shape[0]

def random_MPS (N, phydim, vdim=1):
    mps = []
    for i in range(N):
        if i == 0:
            mps.append (np.random.rand (1, phydim, vdim))
        elif i == N-1:
            mps.append (np.random.rand (vdim, phydim, 1))
        else:
            mps.append (np.random.rand (vdim, phydim, vdim))
    return mps

def compress_MPS (mps, D=sys.maxsize, cutoff=0.):
    N = len(mps)
    #
    #        2 ---
    #            |
    #   R =      o
    #            |
    #        1 ---
    #
    Rs = [None for i in range(N+1)]
    Rs[-1] = np.array([1.]).reshape((1,1))
    for i in range(N-1, 0, -1):
        Rs[i] = ncon([Rs[i+1],mps[i],np.conjugate(mps[i])], ((1,2), (-1,3,1), (-2,3,2)))


    #
    #          2
    #          |
    #   rho =  o
    #          |
    #          1
    #
    rho = ncon([Rs[1],mps[0],np.conjugate(mps[0])], ((1,2), (-1,-2,1), (-3,-4,2)))
    rho = rho.reshape((rho.shape[1], rho.shape[3]))

    #         1
    #         |
    #    0 x--o-- 2
    #
    evals, U = np.linalg.eigh(rho)
    U = U.reshape((1,*U.shape))
    res = [U]

    #
    #        ---- 2
    #        |
    #   L =  |
    #        |
    #        ---- 1
    #
    L = np.array([1.]).reshape((1,1))
    for i in range(1,N):
        #
        #         2---(U)-- -2
        #         |    |
        #   L =  (L)   3
        #         |    |
        #         1---(A)--- -1
        #
        L = ncon([L,mps[i-1],np.conjugate(U)], ((1,2), (1,3,-1), (2,3,-2)))

        #
        #          -- -1
        #          |
        #   A =    |       -2
        #          |        |
        #         (L)--1--(mps)--- -3
        #
        A = ncon([L,mps[i]], ((1,-1), (1,-2,-3)))

        #
        #         -3 --(A)--2--
        #               |     |
        #              -4     |
        #   rho =            (R)
        #              -2     |
        #               |     |
        #         -1 --(A)--1--
        #
        rho = ncon([Rs[i+1],A,np.conjugate(A)], ((1,2), (-1,-2,1), (-3,-4,2)))
        d = rho.shape
        rho = rho.reshape((d[0]*d[1], d[2]*d[3]))

        #         1
        #         |
        #     0 --o-- 2
        #
        evals, U  = np.linalg.eigh(rho)
        # truncate by dimension
        DD = min(D, mps[i].shape[2])
        U = U[:,-DD:]
        evals = evals[-DD:]
        # truncate by cutoff
        iis = (evals > cutoff)
        U = U[:,iis]
        #
        U = U.reshape((d[2],d[3],U.shape[1]))
        res.append(U)
    return res

def sum_mps_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2]+T2.shape[2]))
    res[:T1.shape[0],:,:T1.shape[2]] = T1
    res[T1.shape[0]:,:,T1.shape[2]:] = T2
    return res

def inner_MPS (mps1, mps2):
    res = ncon([mps1[0], np.conjugate(mps2[0])], ((1,2,-1), (1,2,-2)))
    for i in range(1,len(mps1)):
        res = ncon([res,mps1[i],np.conjugate(mps2[i])], ((1,2), (1,3,-1), (2,3,-2)))
    return float(res)

def mps_func_to_mpo (mps):
    check_MPS_links (mps)

    mpo = []
    for A in mps:
        T = np.zeros((A.shape[0],A.shape[1],A.shape[1],A.shape[2]))
        for i in range(A.shape[0]):
            for j in range(A.shape[2]):
                ele = A[i,:,j]
                T[i,:,:,j] = np.diag(ele)
        mpo.append(T)
    L = np.array([1.])
    R = np.array([1.])
    return mpo, L, R

#======================== MPO ========================
# For each tensor, the order of index is (left, i', i, right)
# An MPO also has a left and a right tensor

def check_MPO_links (mpo, L, R):
    assert mpo[0].shape[0] == L.shape[0]
    assert mpo[-1].shape[-1] == R.shape[0]
    for i in range(len(mpo)):
        assert mpo[i].ndim == 4
        if i != 0:
            assert mpo[i-1].shape[-1] == mpo[i].shape[0]

def identity_MPO (N, phydim):
    As = [np.identity(phydim).reshape((1,phydim,phydim,1)) for i in range(N)]
    L = np.ones(1)
    R = np.ones(1)
    return As, L, R

def absort_L (mpo, L):
    mpo = copy.copy(mpo)
    shape = (1, *mpo[0].shape[1:])
    mpo[0] = ncon([mpo[0],L], ((1,-1,-2,-3),(1,))).reshape(shape)
    return mpo

def absort_R (mpo, R):
    mpo = copy.copy(mpo)
    shape = (*mpo[-1].shape[:3], 1)
    mpo[-1] = ncon([mpo[-1],R], ((-1,-2,-3,1),(1,))).reshape(shape)
    return mpo

def absort_LR (mpo, L, R):
    mpo = absort_L (mpo, L)
    mpo = absort_R (mpo, R)
    return mpo

def product_2MPO (mpo1, L1, R1, mpo2, L2, R2):
    mpo1 = copy.copy(mpo1)
    mpo2 = copy.copy(mpo2)
    # Absort R1 into mpo1[-1]
    mpo1 = absort_R (mpo1, R1)
    # Absort L2 into mpo2[0]
    mpo2 = absort_L (mpo2, L2)

    L = L1
    R = R2
    assert type(mpo1) == list and type(mpo2) == list
    # Combine the two MPO
    mpo = mpo1 + mpo2

    return mpo, L, R

def purify_MPO (mpo, L, R, cutoff=0.):
    # Make the MPO like an MPS by splitting the physical indices
    mps = []
    for A in mpo:
        A1, A2 = nput.truncate_svd2 (A, 2, cutoff)
        mps += [A1, A2]

    mps[0] = ncon ([L,mps[0]], ((1,), (1,-1,-2)))
    mps[-1] = ncon ([R,mps[-1]], ((1,), (-1,-2,1)))
    mps[0] = mps[0].reshape((1,*mps[0].shape))
    mps[-1] = mps[-1].reshape((*mps[-1].shape,1))
    return mps

def compress_MPO (mpo, L, R, cutoff):
    # Make the MPO like an MPS by splitting the physical indices
    mps = purify_MPO (mpo, L, R)

    mps2 = compress_MPS (mps, cutoff=cutoff)    # <mps2|mps2> = 1
    c = inner_MPS (mps,mps2)
    mps2[0] *= c

    # Back to MPO
    res = []
    for i in range(0,len(mps),2):
        A = ncon([mps2[i],mps2[i+1]], ((-1,-2,1), (1,-3,-4)))
        res.append(A)

    L = np.array([1.])
    R = np.array([1.])
    return res, L, R

def sum_mpo_tensor (T1, T2):
    assert T1.ndim == T2.ndim == 4
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3]+T2.shape[3]))
    res[:T1.shape[0],:,:,:T1.shape[3]] = T1
    res[T1.shape[0]:,:,:,T1.shape[3]:] = T2
    return res

def sum_2MPO (mpo1, L1, R1, mpo2, L2, R2):
    assert type(mpo1) == list and type(mpo2) == list
    N = len(mpo1)
    assert N == len(mpo2)

    mpo = []
    for n in range(N):
        A = sum_mpo_tensor (mpo1[n], mpo2[n])
        mpo.append(A)

    L = np.concatenate ((L1, L2))
    R = np.concatenate ((R1, R2))
    return mpo, L, R

def inner_MPO (mps1, mps2, mpo, L, R):
    res = L.reshape((1,L.shape[0],1))
    for i in range(len(mps1)):
        res = ncon([res,mps1[i],mpo[i],np.conjugate(mps2[i])], ((1,2,3), (1,4,-1), (2,5,4,-2), (3,5,-3)))
    res = ncon([res, R.reshape((1,R.shape[0],1))], ((1,2,3), (1,2,3)))
    return float(res)

