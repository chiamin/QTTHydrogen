import sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
from ncon import ncon

I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])

def print_bonds_info (T):
    print(T.labels())
    print(T.bonds())

def inds_to_x (inds, rescale=1., shift=0.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift

def mpo_to_mps (qtt):
    p = np.zeros((2,2,2))
    p[0,0,0] = 1.
    p[1,1,1] = 1.
    # Reduce to matrix multiplication
    Ms = []                 # Store the matrices after index collapsing
    N = len(qtt)            # The number of tensors in QTT
    for n in range(N):      # For each tensor
        M = qtt[n]
        M = ncon ([M,p], ((-1,1,2,-3), (1,2,-2)))
        Ms.append(M)
    return Ms

def get_ele_mps (mps, inds):
    # Reduce to matrix multiplication
    N = len(qtt)            # The number of tensors in QTT
    res = mps[0][:,inds[0],:]
    for n in range(1,N):
        res = res @ qtt[n][:,inds[n],:]
    return res    

def get_ele_func (qtt, L, R, inds):
    # Reduce to matrix multiplication
    N = len(qtt)            # The number of tensors in QTT
    res = L
    for n in range(N):      # For each tensor
        ind = inds[n]       # The index number we want to collapse
        M = qtt[n][:,ind,:]
        res = res @ M
    res = res @ R
    return res    

def get_ele_mpo (mpo, L, R, inds):
    N = len(mpo)            # The number of tensors in QTT
    res = L.reshape((1,L.shape[0]))
    for n in range(len(mpo)):
        M = mpo[n]
        M = M[:, inds[n], inds[n], :]
        res = np.dot(res, M)
    res = res @ R.reshape((R.shape[0],1))
    return float(res)

def mpo_contract_all (mpo):
    res = mpo[0]
    for A in mpo[1:]:
        ds1 = [-i for i in range(1,len(res.shape))]
        ds2 = [-i+ds1[-1] for i in range(1,len(A.shape))]
        res = ncon([res,A], ((*ds1,1), (1,*ds2)))
    N = len(mpo)
    #res = np.transpose(res, list(range(0,2*N,2))+list(range(1,2*N,2)))
    #res = res.reshape(res.size//2, res.size//2)
    return res

def contract_L (mpo, L):
    Ltmp = L.reshape((1,*L.shape))
    mpo[0] = ncon ([Ltmp,mpo[0]], ((-1,1), (1,-2,-3,-4)))
    return mpo

def contract_R (mpo, R):
    Rtmp = R.reshape((*R.shape,1))
    mpo[-1] = ncon ([Rtmp,mpo[-1]], ((1,-4), (-1,-2,-3,1)))
    return mpo

def contract_LR (mpo, L, R):
    mpo = contract_L (mpo, L)
    mpo = contract_R (mpo, R)
    return mpo

def sum_mpo_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3]+T2.shape[3]))
    res[:T1.shape[0],:,:,:T1.shape[3]] = T1
    res[T1.shape[0]:,:,:,T1.shape[3]:] = T2
    return res

def sum_mps_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2]+T2.shape[2]))
    res[:T1.shape[0],:,:T1.shape[2]] = T1
    res[T1.shape[0]:,:,T1.shape[2]:] = T2
    return res

def mps_to_uniten (mps):
    res = []
    for i in range(len(mps)):
        A = cytnx.UniTensor (cytnx.from_numpy(mps[i]), rowrank=2)
        A.set_labels(['l','i','r'])
        res.append(A)
    return res

def mpo_to_uniten (mpo,L,R):
    H = []
    for i in range(len(mpo)):
        h = toUniTen(mpo[i])
        h.relabels_(['l','ip','i','r'])
        H.append(h)

    Lr = L.reshape((len(L),1,1))
    Rr = R.reshape((len(R),1,1))
    Lr = toUniTen (Lr)
    Rr = toUniTen (Rr)
    Lr.relabels_(['mid','up','dn'])
    Rr.relabels_(['mid','up','dn'])
    return H, Lr, Rr

def generate_random_MPS_nparray (Nsites, d, D):
    psi = [None for i in range(Nsites)]
    for i in range(Nsites):
        if i == 0:
            A = np.random.rand(1,d,D)
        elif i == (Nsites-1):
            A = np.random.rand(D,d,1)
        else:
            A = np.random.rand(D,d,D)
        psi[i] = A
    c = inner(psi,psi)
    psi[0] /= c**0.5
    return psi

def inner (mps1, mps2):
    res = ncon([mps1[0], np.conjugate(mps2[0])], ((1,2,-1), (1,2,-2)))
    for i in range(1,len(mps1)):
        res = ncon([res,mps1[i],np.conjugate(mps2[i])], ((1,2), (1,3,-1), (2,3,-2)))
    return float(res)

def inner_mpo (mps1, mps2, mpo, L, R):
    res = L.reshape((1,L.shape[0],1))
    for i in range(len(mps1)):
        res = ncon([res,mps1[i],mpo[i],np.conjugate(mps2[i])], ((1,2,3), (1,4,-1), (2,5,4,-2), (3,5,-3)))
    res = ncon([res, R.reshape((1,R.shape[0],1))], ((1,2,3), (1,2,3)))
    return float(res)

def toUniTen (T):
    T = cytnx.from_numpy(T)
    return cytnx.UniTensor (T)

def to_nparray(T):
    if T.is_blockform():
        tmp = cytnx.UniTensor.zeros(T.shape())
        tmp.convert_from(T)
        T = tmp
    return T.get_block().numpy()

def mps_to_nparray (mps):
    res = []
    for i in range(len(mps)):
        if mps[i].is_blockform():
            A = cytnx.UniTensor.zeros(mps[i].shape())
            A.convert_from(mps[i])
            A = A.get_block().numpy()
        else:
            A = mps[i].get_block().numpy()
        res.append(A)
    return res

def truncate_svd2 (T, rowrank, cutoff):
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
    U, S, Vh = np.linalg.svd (T)
    U = U[:,:len(S)]
    Vh = Vh[:len(S),:]
    ii = (S >= cutoff)
    U, S, Vh = U[:,ii], S[ii], Vh[ii,:]

    A = (U*S).reshape(*ds1,-1)
    B = Vh.reshape(-1,*ds2)
    return A, B

def compress_mps (mps, D=sys.maxsize, cutoff=0.):
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

def purify_mpo (mpo, L, R, cutoff=0.):
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
    mps = purify_mpo (mpo, L, R)

    mps2 = compress_mps (mps, cutoff=cutoff)    # <mps2|mps2> = 1
    c = inner(mps,mps2)
    mps2[0] *= c

    # Back to MPO
    res = []
    for i in range(0,len(mps),2):
        A = ncon([mps2[i],mps2[i+1]], ((-1,-2,1), (1,-3,-4)))
        res.append(A)

    L = np.array([1.])
    R = np.array([1.])
    return res, L, R

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

    T = ncon ([T1,T2], ((-1,-3,1,-5), (-2,1,-4,-6)))
    T = T.reshape ((dk1,dipr,di,dk2))
    return T

