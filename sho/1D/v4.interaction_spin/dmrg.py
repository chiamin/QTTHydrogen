import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
import qtt_utility as ut
import MPS_utility as mpsut
import utUtility as utut

class LR_envir_tensors_mpo:
    def __init__ (self, N, L0, R0):
        # Network for computing right environment tensors
        self.LR_env_net = cytnx.Network()
        # Labels with underscore will be contracted
        self.LR_env_net.FromString(["LR: _mid, _up, _dn",
                                    "A: _dn, _i, dn",
                                    "M: _mid, mid, _ip, _i",
                                    "A_Conj: _up, _ip, up",
                                    "TOUT: mid, up, dn"])
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None for i in range(N+1)]
        self.LR[0]  = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]

    # self.LR[i] and self.LR[i+1] are the left and right environments for the ith site
    def update_LR (self, mps1, mps2, mpo, center):
        # Update the left environments
        for p in range(self.centerL, center):
            # Set the network for the left environment tensor on the current site
            '''self.LR_env_net.PutUniTensor("LR", self.LR[p], ['mid','up','dn'])
            self.LR_env_net.PutUniTensor("A", mps1[p], ['l','i','r'])
            self.LR_env_net.PutUniTensor("M", mpo[p], ['l','r','ip','i'])
            self.LR_env_net.PutUniTensor("A_Conj", mps2[p].Dagger(), ['l','i','r'])
            self.LR[p+1] = self.LR_env_net.Launch()
            self.LR[p+1].relabels_(['mid','up','dn'])'''

            A1 = self.LR[p].relabels(['mid','dn','up'],['_mid','_dn','_up'])
            A2 = mps1[p].relabels(['l','i','r'],['_dn','_i','dn'])
            A3 = mpo[p].relabels(['l','r','ip','i'],['_mid','mid','_ip','_i'])
            A4 = mps2[p].Dagger().relabels(['l','i','r'],['_up','_ip','up'])

            tmp = cytnx.Contract(A1,A2)
            tmp = cytnx.Contract(tmp,A3)
            tmp = cytnx.Contract(tmp,A4)

            self.LR[p+1] = tmp
        # Update the right environments
        for p in range(self.centerR, center, -1):
            # Set the network for the right environment tensor on the current site
            '''self.LR_env_net.PutUniTensor("LR", self.LR[p+1], ['mid','up','dn'])
            self.LR_env_net.PutUniTensor("A", mps1[p], ['r','i','l'])
            self.LR_env_net.PutUniTensor("M", mpo[p], ['r','l','ip','i'])
            self.LR_env_net.PutUniTensor("A_Conj", mps2[p].Dagger(), ['r','i','l'])
            self.LR[p] = self.LR_env_net.Launch()
            self.LR[p].relabels_(['mid','up','dn'])'''

            A1 = self.LR[p+1].relabels(['mid','dn','up'],['_mid','_dn','_up'])
            A2 = mps1[p].relabels(['r','i','l'],['_dn','_i','dn'])
            A3 = mpo[p].relabels(['r','l','ip','i'],['_mid','mid','_ip','_i'])
            A4 = mps2[p].Dagger().relabels(['r','i','l'],['_up','_ip','up'])

            tmp = cytnx.Contract(A1,A2)
            tmp = cytnx.Contract(tmp,A3)
            tmp = cytnx.Contract(tmp,A4)
            self.LR[p] = tmp

        self.centerL = self.centerR = center

class LR_envir_tensors_mps:
    def __init__ (self, N, mps1, mps2):
        # Network for computing right environment tensors
        self.env_net = cytnx.Network()
        self.env_net.FromString(["LR: _up, _dn",
                                 "A: _dn, i, dn",
                                 "Adag: _up, i, up",
                                 "TOUT: up, dn"])
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None for i in range(N+1)]
        l1 = mps1[0].bond("l").redirect()
        l2 = mps2[0].bond("l")
        r1 = mps1[-1].bond("r").redirect()
        r2 = mps2[-1].bond("r")
        L0 = cytnx.UniTensor ([l1,l2], labels=['dn','up'])
        R0 = cytnx.UniTensor ([r1,r2], labels=['dn','up'])
        assert np.prod(L0.shape()) == np.prod(R0.shape()) == 1
        L0.at([0,0]).value = 1.
        R0.at([0,0]).value = 1.
        self.LR[0]  = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]

    def update_LR (self, mps1, mps2, center):
        # Update the left environments
        for p in range(self.centerL, center):
            # Set the network for the left environment tensor on the current site
            '''self.env_net.PutUniTensor("LR", self.LR[p], ['up','dn'])
            self.env_net.PutUniTensor("A", mps1[p], ['l','i','r'])
            self.env_net.PutUniTensor("Adag", mps2[p].Dagger(), ['l','i','r'])
            self.LR[p+1] = self.env_net.Launch()'''

            A1 = self.LR[p].relabels(['up','dn'],['_up', '_dn'])
            A2 = mps1[p].relabels(['l','i','r'],['_dn', 'i', 'dn'])
            A3 = mps2[p].Dagger().relabels(['l','i','r'],['_up', 'i', 'up'])

            tmp = cytnx.Contract(A1,A2)
            tmp = cytnx.Contract(tmp,A3)
            self.LR[p+1] = tmp
        # Update the right environments
        for p in range(self.centerR, center, -1):
            # Set the network for the right environment tensor on the current site
            '''self.env_net.PutUniTensor("LR", self.LR[p+1], ['up','dn'])
            self.env_net.PutUniTensor("A", mps1[p], ['r','i','l'])
            self.env_net.PutUniTensor("Adag", mps2[p].Dagger(), ['r','i','l'])
            self.LR[p] = self.env_net.Launch()'''
            A1 = self.LR[p+1].relabels(['up','dn'],['_up', '_dn'])
            A2 = mps1[p].relabels(['r','i','l'],['_dn', 'i', 'dn'])
            A3 = mps2[p].Dagger().relabels(['r','i','l'],['_up', 'i', 'up'])

            tmp = cytnx.Contract(A1,A2)
            tmp = cytnx.Contract(tmp,A3)
            self.LR[p] = tmp
        self.centerL = self.centerR = center

# An effective Hamiltonian must:
# 1. Inherit <cytnx.LinOp> class
# 2. Has a function <matvec> that implements H|psi> operation
class eff_Hamilt (cytnx.LinOp):
    def __init__ (self, L, M1, M2, R):
        # Initialization
        cytnx.LinOp.__init__(self,"mv", 0)

        # Define network for H|psi> operation
        self.anet = cytnx.Network()
        self.anet.FromString(["psi: ldn, i1, i2, rdn",
                              "L: l, ldn, lup",
                              "R: r, rdn, rup",
                              "M1: l, i1, ip1, mid",
                              "M2: mid, ip2, i2, r",
                              "TOUT: lup, ip1, ip2, rup"])
        self.anet.PutUniTensor("L", L, ["mid","dn","up"])
        self.anet.PutUniTensor("M1", M1, ["l","i","ip","r"])
        self.anet.PutUniTensor("M2", M2, ["l","ip","i","r"])
        self.anet.PutUniTensor("R", R, ["mid","dn","up"])

        self.L = L.relabels(['mid','dn','up'], ['l','ldn','lup'])
        self.M1 = M1.relabels(['l','ip','i','r'], ['l','ip1','i1','mid'])
        self.M2 = M2.relabels(['l','ip','i','r'], ['mid','ip2','i2','r'])
        self.R = R.relabels(['mid','dn','up'], ['r','rdn','rup'])


        # For excited states
        self.anet2 = cytnx.Network()
        self.anet2.FromString(["A1: lup, i1, _",
                               "A2: _, i2, rup",
                               "L: ldn, lup",
                               "R: rdn, rup",
                               "TOUT: ldn, i1, i2, rdn"])
        self.ortho = []
        self.ortho_w = []

    def add_orthogonal (self, L, orthoA1, orthoA2, R, weight):
        self.anet2.PutUniTensor("L", L, ["dn","up"])
        self.anet2.PutUniTensor("R", R, ["dn","up"])
        self.anet2.PutUniTensor("A1", orthoA1, ["l","i","r"])
        self.anet2.PutUniTensor("A2", orthoA2, ["l","i","r"])
        out = self.anet2.Launch()
        out.relabels_(['l','i1','i2','r'])
        self.ortho.append(out)
        self.ortho_w.append(weight)

    # Define H|psi> operation
    def matvec (self, v):
        psi = v.relabels(['l','i1','i2','r'],['ldn','i1','i2','rdn'])

        tmp = cytnx.Contract (self.L, psi)
        tmp = cytnx.Contract (tmp, self.M1)
        tmp = cytnx.Contract (tmp, self.M2)
        tmp = cytnx.Contract (tmp, self.R)


        out = tmp
        out.relabels_(['lup', 'ip1', 'ip2', 'rup'],['l','i1','i2','r'])

        '''self.anet.PutUniTensor("psi",v,['l','i1','i2','r'])
        out = self.anet.Launch()
        out.set_labels(['l','i1','i2','r'])   # Make sure the input labels match the output labels'''

        for j in range(len(self.ortho)):
            ortho = self.ortho[j]
            overlap = cytnx.Contract (ortho, v)
            out += self.ortho_w[j] * overlap.item() * ortho

        return out

def dmrg (psi, H, L0, R0, maxdims, cutoff, maxIter=2, ortho_mpss=[], weights=[], verbose=True):
    # Check the MPS and the MPO
    assert (len(psi) == len(H))
    mpsut.check_mpo_bonds (H, L0, R0)
    mpsut.check_mps_bonds (psi)

    # Define the links to update for a sweep
    # First do a right-to-left and then a left-to-right sweep
    Nsites = len(psi)
    ranges = [range(Nsites-2,-1,-1), range(Nsites-1)]

    # For printing information
    label = ["[r->l]", "[l->r]"]


    # Get the environment tensors
    LR = LR_envir_tensors_mpo (Nsites, L0, R0)
    LR.update_LR (psi, psi, H, Nsites-1)

    LR_ortho = []
    for omps in ortho_mpss:
        lr = LR_envir_tensors_mps (Nsites, psi, omps)
        lr.update_LR (psi, omps, Nsites-1)
        LR_ortho.append (lr)
    
    ens = []
    for k in range(len(maxdims)):                                                            # For each sweep
        chi = maxdims[k]                                                                     # Read bond dimension
        print ('Sweep',k,', chi='+str(chi))
        for lr in [0,1]:
            for p in ranges[lr]:                                                             # For each bond
                M1, M2 = H[p], H[p+1]
                # Compute the current psi
                A1 = psi[p].relabels(['l','i','r'], ['l','i1','_'])
                A2 = psi[p+1].relabels(['l','i','r'],['_','i2','r'])
                phi = cytnx.Contract (A1, A2)

                # Define the effective Hamiltonian
                effH = eff_Hamilt (LR[p], M1, M2, LR[p+2])

                # orthogonal MPS
                for j in range(len(ortho_mpss)):
                    omps = ortho_mpss[j]
                    weight = weights[j]
                    oLR = LR_ortho[j]
                    effH.add_orthogonal (oLR[p], omps[p], omps[p+1], oLR[p+2], weight)

                # Find the ground state for the current bond
                enT, phi = cytnx.linalg.Lanczos (effH, phi, method="Gnd", Maxiter=maxIter, CvgCrit=100000)
                en = enT.item()       # Tensor to number

                # SVD and truncate the wavefunction psi
                phi.set_rowrank_(2)
                s, u, vt = cytnx.linalg.Svd_truncate(phi, keepdim=chi, err=cutoff)
                # s has labels _aux_L and _aux_R

                # Setting psi[p] = u, psi[p+1] = vt
                psi[p] = u
                psi[p+1] = vt
                
                # Normalize the singular values
                s = s/s.Norm().item()

                if lr == 0:
                    # Absorb s into next neighbor
                    psi[p] = cytnx.Contract(psi[p],s)

                    psi[p].relabels_(['l', 'i1', '_aux_R'],['l','i','r'])
                    psi[p+1].relabels_(['_aux_R', 'i2', 'r'],['l','i','r'])

                    LR.update_LR (psi, psi, H, p)
                    for j in range(len(ortho_mpss)):
                        LR_ortho[j].update_LR (psi, ortho_mpss[j], p)
                if lr == 1:
                    # Absorb s into next neighbor
                    psi[p+1] = cytnx.Contract(s,psi[p+1])

                    psi[p].relabels_(['l', 'i1', '_aux_L'],['l','i','r'])
                    psi[p+1].relabels_(['_aux_L', 'i2', 'r'],['l','i','r'])

                    LR.update_LR (psi, psi, H, p+1)
                    for j in range(len(ortho_mpss)):
                        LR_ortho[j].update_LR (psi, ortho_mpss[j], p+1)
            if verbose:
                print ('\t',label[lr],'energy =',en)

        ens.append(en);
    return psi, ens

