import sys, copy
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import time

def gradient (mps, mpo, LR, p):
    #  ----
    #  |  |---------O--- -1
    #  |  |    1    |
    #  |  |        -2
    #  |  |--- -3
    #  |  |
    #  |  |
    #  |  |--- -4
    #  ----
    tmp = ncon((LR[p-1],np.conj(mps[p])),((1,-3,-4),(1,-2,-1)))
    #  ----
    #  |  |---------O--- -1
    #  |  |         |
    #  |  |       1 |
    #  |  |---------O--- -2
    #  |  |    2    |
    #  |  |        -3
    #  |  |--- -4
    #  ----
    tmp = ncon((tmp,mpo[p]),((-1,1,2,-4),(2,1,-3,-2)))
    #  ----                   ----
    #  |  |---------O---------|  |
    #  |  |         |    1    |  |
    #  |  |         |         |  |
    #  |  |---------O---------|  |
    #  |  |         |    2    |  |
    #  |  |        -2         |  |
    #  |  |---- -1     -3 ----|  |
    #  ----                   ----
    tmp = ncon((tmp,LR[p+1]),((1,2,-2,-1),(1,2,-3)))
    return tmp

def get_en (mps, mpo, LR, p):
    npmps.check_canonical (mps, p)
    LR.update_LR (mps, mps, mpo, p)
    dA = gradient (mps, mpo, LR, p)
    en = np.inner (mps[p].reshape(-1), dA.reshape(-1))
    return en

def gradient_descent (mps, mpo, gamma, maxdim=100000000, cutoff=1e-16):
    assert len(mps) == len(mpo)
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    #npmps.check_canonical (mps, 0)

    mps = copy.copy(mps)

    N = len(mps)
    LR = dmrg.LR_envir_tensors_mpo (N)

    sites = [range(N-1), range(N-1,0,-1)]
    gd_time, update_time = 0, 0
    for lr in [0,1]:
        for p in sites[lr]:
            start = time.time()
            LR.update_LR (mps, mps, mpo, p)
            dA = gradient (mps, mpo, LR, p)
            dA = np.conj(dA)
            end = time.time()
            gd_time += end - start

            #npmps.check_canonical (mps, p)
            en = np.inner (mps[p].reshape(-1), dA.reshape(-1))

            '''for gamma in np.linspace(0,0.000001,20):
                mpst = copy.copy(mps)
                A = mps[p] - gamma * dA
                A /= np.linalg.norm(A)
                mpst[p] = A
                ent = get_en (mpst, mpo, LR, p)
                print('\ten(gamma) =',ent, gamma)
            exit()'''

            start = time.time()
            #gamma = np.linalg.norm(dA) * 1e-12
            A = mps[p] - gamma * dA
            #print('p',p,np.linalg.norm(dA),en)
            A /= np.linalg.norm(A)
            # update sites p and (p+1 or p-1)
            mps[p] = A
            mps = dmrg.orthogonalize_MPS_tensor (mps, p, toRight=(lr==0), maxdim=maxdim, cutoff=cutoff)
            end = time.time()
            update_time += end - start
    print('GD gradient time',end-start)
    print('GD update time',end-start)

    # Compute energy
    #npmps.check_canonical (mps, 0)
    start = time.time()
    LR.update_LR (mps, mps, mpo, 0)
    dA = gradient (mps, mpo, LR, 0)
    en = np.inner (mps[0].reshape(-1), dA.reshape(-1))
    #print('gg',p,en,npmps.inner_MPO(mps,mps,mpo,L,R))
    end = time.time()
    print('GD energy time',end-start)

    return mps, en
