import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy as sp
from ncon import ncon
import npmps
import lanczos

class LR_envir_tensors_mpo:
    def __init__ (self, N, L0, R0):
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None for i in range(N+1)]
        self.LR[0]  = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]

    # self.LR[i] and self.LR[i+1] are the left and right environments for the ith site
    # MPS tensor indices = (l,i,r)
    # MPO tensor indices = (l,ip,i,r)
    # Left or right environment tensor = (up, mid, down)
    def update_LR (self, mps1, mps2, mpo, centerL, centerR=None):
        if centerR == None:
            centerR = centerL
        # Update the left environments
        for p in range(self.centerL, centerL):
            #  ----
            #  |  |--- -1
            #  |  |
            #  |  |--- -2
            #  |  |        -3
            #  |  |         |
            #  |  |---------O--- -4
            #  ----    1
            tmp = ncon((self.LR[p],mps1[p]),((-1,-2,1),(1,-3,-4)))
            #  ----
            #  |  |--- -1
            #  |  |        -2
            #  |  |         |
            #  |  |---------O--- -3
            #  |  |    1    |
            #  |  |         | 2
            #  |  |---------O--- -4
            #  ----
            tmp = ncon((tmp,mpo[p]),((-1,1,2,-4),(1,-2,2,-3)))
            #  ----
            #  |  |---------O--- -1
            #  |  |    1    |
            #  |  |         | 2
            #  |  |---------O--- -2
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -3
            #  ----
            tmp = ncon((tmp,np.conj(mps2[p])),((1,2,-2,-3),(1,2,-1)))

            self.LR[p+1] = tmp
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            #                 ----
            #           -1 ---|  |
            #                 |  |
            #           -2 ---|  |
            #       -3        |  |
            #        |        |  |
            #  -4 ---O--------|  |
            #             1   ----
            tmp = ncon((self.LR[p+1],mps1[p]),((-1,-2,1),(-4,-3,1)))
            #                 ----
            #           -1 ---|  |
            #       -2        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #        |    1   |  |
            #        | 2      |  |
            #  -4 ---O--------|  |
            #                 ----
            tmp = ncon((tmp,mpo[p]),((-1,1,2,-4),(-3,-2,2,1)))
            #                 ----
            #  -1 ---O--------|  |
            #        |    1   |  |
            #        | 2      |  |
            #  -2 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #                 ----
            tmp = ncon((tmp,np.conj(mps2[p])),((1,2,-2,-3),(-1,2,1)))
            self.LR[p] = tmp

        self.centerL = centerL
        self.centerR = centerR


# An effective Hamiltonian must:
# 1. Inherit <cytnx.LinOp> class
# 2. Has a function <matvec> that implements H|psi> operation
class eff_Hamilt_2sites (sp.sparse.linalg.LinearOperator):
    def __init__ (self, L, M1, M2, R, dtype=float):
        self.L = L
        self.R = R
        self.M1 = M1
        self.M2 = M2

        dim = L.shape[0] * M1.shape[1] * M2.shape[1] * R.shape[0]
        self.shape = (dim,dim)
        self.dtype = np.dtype(dtype)
        self.vshape = (L.shape[0], M1.shape[1], M2.shape[1], R.shape[0])

    # Define H|v> operation
    #
    #              2    3
    # |v> =        |____|
    #        1 ---(______)--- 4
    def _matvec (self, v):
        v = v.reshape(self.vshape)
        #  ----
        #  |  |--- -1
        #  |  |
        #  |  |--- -2
        #  |  |        -3   -4
        #  |  |         |____|
        #  |  |--------(______)--- -5
        #  ----    1
        tmp = ncon((self.L, v),((-1,-2,1),(1,-3,-4,-5)))
        #  ----
        #  |  |--- -1
        #  |  |        -2
        #  |  |         |
        #  |  |---------O--- -3
        #  |  |    1    |        -4
        #  |  |       2 |_________|
        #  |  |--------(__________)--- -5
        #  ----
        tmp = ncon((tmp,self.M1),((-1,1,2,-4,-5),(1,-2,2,-3)))
        #  ----
        #  |  |--- -1
        #  |  |        -2        -3
        #  |  |         |         |
        #  |  |---------O---------O--- -4
        #  |  |         |    1    |
        #  |  |         |_________| 2
        #  |  |--------(__________)--- -5
        #  ----
        tmp = ncon((tmp,self.M2),((-1,-2,1,2,-5),(1,-3,2,-4)))
        #  ----                           ----
        #  |  |--- -1               -4 ---|  |
        #  |  |        -2        -3       |  |
        #  |  |         |         |       |  |
        #  |  |---------O---------O-------|  |
        #  |  |         |         |   1   |  |
        #  |  |         |_________|       |  |
        #  |  |--------(__________)-------|  |
        #  ----                       2   ----
        tmp = ncon((tmp,self.R),((-1,-2,-3,1,2),(-4,1,2)))

        return tmp

def dmrg_2sites (psi, H, L0, R0, maxdims, cutoff, maxIter=10000, krylovDim=20, tol=1e-16, ortho_mpss=[], weights=[], verbose=False):
    # Check the MPS and the MPO
    assert (len(psi) == len(H))
    npmps.check_MPO_links (H, L0, R0)
    npmps.check_MPS_links (psi)

    dL, dR = len(L0), len(R0)
    L0 = L0.reshape((1,dL,1))
    R0 = R0.reshape((1,dL,1))

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    Nsites = len(psi)
    ranges = [range(Nsites-1), range(Nsites-2,-1,-1)]

    # Get the environment tensors
    LR = LR_envir_tensors_mpo (Nsites, L0, R0)
    LR.update_LR (psi, psi, H, 0)

    ens, terrs = [], []
    N_update = len(ranges[0]) + len(ranges[1])
    for k in range(len(maxdims)):                                                            # For each sweep
        chi = maxdims[k]                                                                     # Read bond dimension
        terr = 0.
        for lr in [0,1]:
            for p in ranges[lr]:
                print(p)
                #
                #        -2    -3
                #         |  1  |
                #   -1 ---O-----O--- -4
                phi = ncon((psi[p],psi[p+1]),((-1,-2,1),(1,-3,-4)))
                dims = phi.shape
                phi = phi.reshape(-1)

                # Define the effective Hamiltonian
                effH = eff_Hamilt_2sites (LR[p], H[p], H[p+1], LR[p+2])

                # Krylov subspace dimension
                krylovDim = min(krylovDim, effH.shape[0]-1)

                # Find the ground state for the current bond
                en, phi = sp.sparse.linalg.eigsh (effH, k=1, v0=phi, which="SA", return_eigenvectors=True, ncv=krylovDim, maxiter=maxIter, tol=tol)
                en = en[0]
                phi = phi[:,0].reshape(dims)
                norm = np.linalg.norm(phi)
                phi /= norm

                # SVD
                toRight = (lr == 0)
                psi[p], psi[p+1], err = npmps.truncate_svd2 (phi, 2, toRight, cutoff=cutoff)

                if lr == 0:
                    LR.update_LR (psi, psi, H, p+1)
                else:
                    LR.update_LR (psi, psi, H, p)

                terr += err

        if verbose:
            print ('Sweep',k,', chi='+str(chi))
            print ('\t','energy =',en, terr)

        ens.append(en);
        terrs.append (terr/N_update)
    return psi, ens, terrs
