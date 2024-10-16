import numpy as np
from ncon import ncon
import sys, copy

def truncate_svd2 (T, rowrank, toRight, cutoff=0.):
    ds = T.shape
    d1, d2 = 1, 1
    ds1, ds2 = [],[]
    for i in range(rowrank):
        d1 *= ds[i]
        ds1.append(ds[i])
    for i in range(rowrank,len(ds)):
        d2 *= ds[i]
        ds2.append(ds[i])
    T = T.reshape((d1,d2))
    U, S, Vh = np.linalg.svd (T, full_matrices=False)

    ii = (S < cutoff)
    terr = np.sum(S[ii])

    ii = (S >= cutoff)
    U, S, Vh = U[:,ii], S[ii], Vh[ii,:]

    if toRight:
        A = U.reshape(*ds1,-1)
        B = (S[:,np.newaxis] * Vh).reshape(-1,*ds2)
    else:
        A = (U*S).reshape(*ds1,-1)
        B = Vh.reshape(-1,*ds2)
    return A, B, terr

#       |
#      (A)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#      (B)
#       |
#
def applyLocal_mpo_tensor (h, A=None, B=None):
    re = ncon([h,A], ((-1,-2,1,-4), (1,-3)))
    re = ncon([re,B], ((-1,1,-3,-4), (-2,1)))
    return re

#       |
#      (A)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#      (B)
#       |
#
def applyLocal_mpo (H, A=None, B=None):
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], A, B)
        res.append(h)
    return res

#       |
#      (U)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#     (Udag)
#       |
#
def applyLocalRot_mpo (H, U):
    Udag = np.conjugate(np.transpose(U))
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], U, Udag)
        res.append(h)
    return res

#       |
#     (op)
#       |
#   ---(A)---
#
def applyLocal_mps (mps, op):
    res = []
    for i in range(len(mps)):
        A = mps[i]
        A = ncon([A,op], ((-1,1,-3), (-2,1)))
        res.append(A)
    return res

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

def to_canonical_form (mps, oc):
    mps = copy.copy(mps)
    for i in range(oc):
        AA = ncon([mps[i],mps[i+1]], ((-1,-2,1),(1,-3,-4)))
        mps[i], mps[i+1] = truncate_svd2 (AA, 2, toRight=True)
    for i in range(len(mps)-1,oc,-1):
        AA = ncon([mps[i-1],mps[i]], ((-1,-2,1),(1,-3,-4)))
        mps[i-1], mps[i] = truncate_svd2 (AA, 2, toRight=False)
    return mps

# !!! Not yet finished
def compress_MPS_center0 (mps, D=sys.maxsize, cutoff=0.):
    N = len(mps)
    #
    #         --- 1
    #         |
    #   L =   o
    #         |
    #         --- 2
    #
    Ls = [None for i in range(N+1)]
    L = np.ones((1,1))
    for i in range(N-1):
        #            3
        #         ------o--- -1
        #         |     |
        #   L =   o     | 2
        #         |  1  |
        #         ------o--- -2
        #
        L = ncon([L,mps[i],np.conj(mps[i])], ((1,3), (-1,2,-2), (3,2,-1)))
        Ls[i] = L

    #            2
    #         --------o--- -1
    #         |       |
    #         |      -2
    #   rho = L
    #         |      -3
    #         |  1    |
    #         --------o--- -4
    d = mps[0].shape[1]
    rho = ncon([L,mps[-1],np.conj(mps[-1])], ((1,2), (1,-3,-4), (2,-2,-1)))
    rho = rho.reshape((d,d))

    #                    (phys)                    (phys)
    #                       1                        2            2
    #                       |     transpose          |            |
    #    U = (virtaul) 2 ---o   =============>  1 ---o   =>  1 ---o---x 3
    evals, U = np.linalg.eigh(rho)
    A = U.transpose().reshape((*U.shape,1))

    res = [None for i in range(N)]
    res[-1] = A

    #
    #        1 ----
    #             |
    #   R =       |
    #             |
    #        2 ----
    #
    R = np.ones((1,1))
    for i in range(N-2,-1,-1):
        #
        #         2---(U)-- -2
        #         |    |
        #   R =  (L)   3
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
    assert len(mps1) == len(mps2)
    res = ncon([mps1[0], np.conjugate(mps2[0])], ((1,2,-1), (1,2,-2)))
    for i in range(1,len(mps1)):
        res = ncon([res,mps1[i],np.conjugate(mps2[i])], ((1,2), (1,3,-1), (2,3,-2)))
    return float(res)

def mps_to_mpo (mps):
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

def normalize_by_integral (mps, x1, x2):
    mps = copy.copy(mps)
    c = inner_MPS (mps, mps)
    mps[0] /= c**0.5

    N = len(mps)
    Ndx = 2**N-1
    dx = (x2-x1)/Ndx

    mps[0] /= dx**0.5
    return mps

#======================== MPO ========================
# For each tensor, the order of index is (left, i', i, right)
# An MPO also has a left and a right tensor

def check_MPO_links (mpo, L, R):
    assert L.ndim == R.ndim == 1
    assert mpo[0].shape[0] == len(L)
    assert mpo[-1].shape[-1] == len(R)
    for i in range(len(mpo)):
        assert mpo[i].ndim == 4
        if i != 0:
            assert mpo[i-1].shape[-1] == mpo[i].shape[0]

def apply_mpo (mpo, L, R, mps):
    assert len(mpo) == len(mps)
    check_MPO_links(mpo, L, R)
    check_MPS_links(mps)

    mpo = copy.copy(mpo)
    mpo = absort_LR (mpo, L, R)

    A1 = ncon([mps[0], mpo[0]], ((-1,1,-4),(-2,-3,1,-5)))
    dl1,dl2,di,dr1,dr2 = A1.shape
    A1 = A1.reshape((1,di,dr1,dr2))
    res = []
    for i in range(1,len(mps)):
        A2 = ncon([mps[i], mpo[i]], ((-1,1,-4),(-2,-3,1,-5)))
        AA = ncon([A1,A2], ((-1,-2,1,2),(1,2,-3,-4,-5)))
        dl,di1,di2,dr1,dr2 = AA.shape
        AA = AA.reshape((dl*di1, di2*dr1*dr2))

        U, S, Vh = np.linalg.svd (AA, full_matrices=False)
        A = (U*S).reshape((dl,di1,-1))
        A1 = Vh.reshape((-1,di2,dr1,dr2))

        res.append(A)
    dl1,di,dr1,dr2 = A1.shape
    A = A1.reshape((dl1,di,1))
    res.append(A)
    return res

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

def direct_product_2MPO (mpo1, L1, R1, mpo2, L2, R2):
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
        A1, A2 = truncate_svd2 (A, 2, cutoff)
        mps += [A1, A2]

    mps[0] = ncon ([L,mps[0]], ((1,), (1,-1,-2)))
    mps[-1] = ncon ([R,mps[-1]], ((1,), (-1,-2,1)))
    mps[0] = mps[0].reshape((1,*mps[0].shape))
    mps[-1] = mps[-1].reshape((*mps[-1].shape,1))
    return mps

def compress_mpo (mpo, L, R, cutoff):
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
    assert len(mps1) == len(mps2) == len(mpo)
    res = L.reshape((1,L.shape[0],1))
    for i in range(len(mps1)):
        res = ncon([res,mps1[i],mpo[i],np.conjugate(mps2[i])], ((1,2,3), (1,4,-1), (2,5,4,-2), (3,5,-3)))
    res = ncon([res, R.reshape((1,R.shape[0],1))], ((1,2,3), (1,2,3)))
    return float(res)

def mpo_contract_all (mpo, L, R):
    mpo = absort_LR (mpo, L, R)
    res = mpo[0]
    for A in mpo[1:]:
        ds1 = [-i for i in range(1,len(res.shape))]
        ds2 = [-i+ds1[-1] for i in range(1,len(A.shape))]
        res = ncon([res,A], ((*ds1,1), (1,*ds2)))
    N = len(mpo)
    return res

# MPO tensor:
#                 (ipr)                   (ipr)                        (ipr)
#                   1                       0                            1
#                   |                       |                            |
#         (k1)  0 --o-- 3 (k2)              o-- 2 (k)           (k)  0 --o
#                   |                       |                            |
#                   2                       1                            2
#                  (i)                     (i)                          (i)
#
#
#                   2                       0                            2
#                   |                       |                            |
#           T1  0 --o-- 4                   o-- 2                    0 --o
#                   |                       |                            |
#           T2  1 --o-- 5                   o-- 3                    1 --o
#                   |                       |                            |
#                   3                       1                            3
#
#
#                   1                       0                           1
#                   |                       |                           |
#               0 --o-- 2                   o-- 1                   0 --o
#
def prod_mpo_tensor (T1, T2):
    di = T2.shape[2]
    dipr = T1.shape[1]
    dk1 = T1.shape[0] * T2.shape[0]
    dk2 = T1.shape[3] * T2.shape[3]

    T = ncon ([T1,T1], ((-1,-3,1,-5), (-2,1,-4,-6)))
    T = T.reshape ((dk1,dipr,di,dk2))
    return T

def H1_to_two_particles (H1, L1, R1):
    I = np.array([[1.,0.],[0.,1.]])

    N = len(H1)
    H = []
    for i in range(N):
        HIi = ncon([H1[i],I], ((-1,-2,-4,-6), (-3,-5)))
        IHi = ncon([H1[i],I], ((-1,-3,-5,-6), (-2,-4)))
        d = H1[i].shape
        HIi = HIi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        IHi = IHi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        H.append (sum_mpo_tensor (HIi, IHi))

    L = np.append(L1,L1)
    R = np.append(R1,R1)
    return H, L, R

def get_H_2D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = identity_MPO (N, 2)
    H1, L1, R1 = direct_product_2MPO (H_1D, L_1D, R_1D, H_I, L_I, R_I)
    H2, L2, R2 = direct_product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H, L, R = sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R


def get_H_3D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H_2I, L_2I, R_2I = npmps.identity_MPO (2*N, 2)
    H1, L1, R1 = direct_product_2MPO (H_1D, L_1D, R_1D, H_2I, L_2I, R_2I)
    H2, L2, R2 = direct_product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H2, L2, R2 = direct_product_2MPO (H2, L2, R2, H_I, L_I, R_I)
    H3, L3, R3 = direct_product_2MPO (H_2I, L_2I, R_2I, H_1D, L_1D, R_1D)
    H, L, R = sum_2MPO (H1, L1, R1, H2, L2, R2)
    H, L, R = sum_2MPO (H, L, R, H3, L3, R3)
    return H, L, R

