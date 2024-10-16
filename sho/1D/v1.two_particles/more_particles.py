import dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import two_particles as tw
import utUtility as utut
import MPS_utility as mpsut
import qn_utility as qnut

def H_4particles (H1, L1, R1):
    H1 = copy.copy(H1)
    H1 = ut.contract_LR (H1, L1, R1)

    I = np.array([[1.,0.],[0.,1.]]).reshape((1,2,2,1))
    N = len(H1)
    I1 = [I for i in range(N)]

    H4_1 = H1 + I1 + I1 + I1
    H4_2 = I1 + H1 + I1 + I1
    H4_3 = I1 + I1 + H1 + I1
    H4_4 = I1 + I1 + I1 + H1

    H = []
    for i in range(len(H4_1)):
        A2 = ut.sum_mpo_tensor (H4_1[i], H4_2[i])
        A3 = ut.sum_mpo_tensor (A2, H4_3[i])
        A4 = ut.sum_mpo_tensor (A3, H4_4[i])
        H.append (A4)

    L = np.array([1.,1.,1.,1.])
    R = np.array([1.,1.,1.,1.])

    return H, L, R


# Get the local quantum number location by the site index (after duplication)
# N is the total number of sites for all the particles
# N1 is the number of sites for single particle
def get_qn_table (N, N1):
    # Get the local quantum number location by the site index (after duplication)
    # qn_loc[site] = qn location in qns
    qn_loc = dict()
    qn_i = 0
    new_i = N1
    for i in range(N1,N-N1):        # For each site in between N1 and N-N1
        qn_loc[new_i] = qn_i        # new_i is the site index after duplication
        qn_loc[new_i+1] = qn_i      # The two duplicated sites have the same local quantum number
        new_i += 2
        qn_i += 1
    return qn_loc

# N is the total number of sites for all the particles
# N1 is the number of sites for single particle
# Each site between N1 and N-N1 will be duplicated to two sites
#
# Return the physical bonds before duplications
def make_physical_bonds (N, N1):
    Nqn = N - 2*N1    # The number of local quantum numbers
    symms = [cytnx.Symmetry.Zn(2)] * Nqn

    iis = []
    for i in range(N):
        qns = [0] * Nqn

        # Not to be duplicated sites
        if i < N1 or i >= 3*N1:
            ii = cytnx.Bond (cytnx.BD_IN, [qns], [2], symms)
        else:
            iqn = i - N1
            qns1 = [0] * Nqn
            qns1[iqn] = 1
            ii = cytnx.Bond (cytnx.BD_IN, [qns, qns1], [1,1], symms)
        iis.append(ii)
    return iis

# All local quantum numbers are 0
def qns0 (N, N1):
    Nqn = N - 2*N1    # The number of local quantum numbers
    return [0] * Nqn

# Make a bond with all local quantum numbers 0
def make_qn0_bond (dim, N, N1):
    Nqn = N - 2*N1
    qns = [0] * Nqn
    symms = [cytnx.Symmetry.Zn(2)] * Nqn
    ii = cytnx.Bond(cytnx.BD_IN, [qns], [dim], symms)
    return ii

def split_physical_indices (H, N1):
    N = len(H)
    iis = make_physical_bonds (N, N1)

    res = []
    for i in range(len(H)):
        dimL = H[i].shape[0]
        dimR = H[i].shape[-1]
        iiL = make_qn0_bond (dimL, N, N1)
        iiR = make_qn0_bond (dimR, N, N1).redirect()
        ii = iis[i]
        iip = ii.redirect()

        if i < N1 or i >= N-N1:
            uA = ut.toUniTen(H[i])
            qnA = cytnx.UniTensor ([iiL,ii,iip,iiR], labels=['l','ip','i','r'], name='s'+str(i))
            qnA.convert_from (uA)
            res.append(qnA)
        else:
            # Duplicate the physical indices
            split = np.zeros((2,2,2))
            split[0,0,0] = 1.
            split[1,1,1] = 1.
            A = ncon([H[i],split,split], ((-1,1,2,-6), (1,-2,-4), (2,-3,-5)))
            #         2   4
            #       __|___|__
            #  1 --(____A____)-- 6
            #         |   |
            #         3   5

            # Assign local quantum number for each duplicated site
            uA = ut.toUniTen(A)
            qnA = cytnx.UniTensor ([iiL,ii,iip,ii,iip,iiR], labels=['l','ip1','i1','ip2','i2','r'])
            qnA.convert_from (uA)

            # Split to two site
            # 1. SVD
            qnA.set_rowrank_(3)
            s, A1, A2 = cytnx.linalg.Svd_truncate (qnA, keepdim=max(dimL,dimR)*4, err=1e-12)
            A1.relabels_(['ip1','i1','_aux_L'],['ip','i','r'])
            # 2. Absort s to A2
            A2 = cytnx.Contract(s,A2)
            A2.relabels_(['ip2','i2','_aux_L'],['ip','i','l'])
            A1.set_name('s'+str(i)+'_1')
            A2.set_name('s'+str(i)+'_2')

            res.append(A1)
            res.append(A2)

    # L and R
    assert L.ndim == 1
    assert R.ndim == 1
    dimL, dimR = L.shape[0], R.shape[0]
    iiL = make_qn0_bond (dimL, N, N1)
    iiR = make_qn0_bond (dimR, N, N1)
    ii0 = make_qn0_bond (1, N, N1)
    ii0p = ii0.redirect()

    qnL = cytnx.UniTensor ([iiL.redirect(), ii0p, ii0], labels=['mid','dn','up'])
    qnR = cytnx.UniTensor ([iiR, ii0, ii0p], labels=['mid','dn','up'])
    uL = ut.toUniTen (L.reshape(1,dimL,1))
    uR = ut.toUniTen (R.reshape(1,dimR,1))
    qnL.convert_from(uL)
    qnR.convert_from(uR)

    return res, qnL, qnR

def make_init_state_bk (N, N1):
    iis = make_physical_bonds (N, N1)
    iiL = make_qn0_bond (1, N, N1)
    iiR = iiL.redirect()

    res = []
    for i in range(N):
        ii = iis[i]
        if i < N1 or i >= N-N1:
            A = cytnx.UniTensor ([iiL,ii,iiR], labels=['l','i','r'])
            A.at([0,0,0]).value = 1/2**0.5
            A.at([0,1,0]).value = 1/2**0.5
            res.append(A)
        else:
            A = cytnx.UniTensor ([iiL,ii,ii,iiR], labels=['l','i1','i2','r'])
            A.at([0,0,0,0]).value = 1/2**0.5
            A.at([0,1,1,0]).value = 1/2**0.5
            A1, A2 = utut.decompose_tensor (A, rowrank=2, cutoff=1e-14)
            A1.relabels_(['i1','aux'],['i','r'])
            A2.relabels_(['i2','aux'],['i','l'])
            res.append(A1)
            res.append(A2)
    return res

def make_init_state_bk2 (N, N1, D=1):
    iis = make_physical_bonds (N, N1)
    iiL0 = make_qn0_bond (1, N, N1)
    iiR0 = iiL0.redirect()
    iiL = make_qn0_bond (D, N, N1)
    iiR = iiL.redirect()

    res = []
    for i in range(N):
        ii = iis[i]
        if i == 0:
            A = cytnx.UniTensor ([iiL0,ii,iiR], labels=['l','i','r'])
            cytnx.random.uniform_(A, low=-1., high=1.)
            res.append(A)
        elif i == N-1:
            A = cytnx.UniTensor ([iiL,ii,iiR0], labels=['l','i','r'])
            cytnx.random.uniform_(A, low=-1., high=1.)
            res.append(A)
        elif i < N1 or i >= N-N1:
            A = cytnx.UniTensor ([iiL,ii,iiR], labels=['l','i','r'])
            cytnx.random.uniform_(A, low=-1., high=1.)
            res.append(A)
        else:
            A = cytnx.UniTensor ([iiL,ii,ii,iiR], labels=['l','i1','i2','r'])
            cytnx.random.uniform_(A, low=-1., high=1.)
            A1, A2 = utut.decompose_tensor (A, rowrank=2, cutoff=1e-14)
            A1.relabels_(['i1','aux'],['i','r'])
            A2.relabels_(['i2','aux'],['i','l'])
            res.append(A1)
            res.append(A2)
    mpsut.check_mps_bonds (res)
    norm = mpsut.inner(res,res)
    res[0] *= 1./norm**0.5
    return res

# iis are the incoming physical bonds
def make_init_state (iis, N, N1):

    res = []
    for i in range(len(iis)):
        ii = iis[i]
        if i == 0:
            iiL = make_qn0_bond (1, N, N1)
            iiR = make_qn0_bond (2, N, N1).redirect()
        elif i == N-1:
            iiL = make_qn0_bond (2, N, N1)
            iiR = make_qn0_bond (1, N, N1).redirect()
        else:
            iiL = make_qn0_bond (2, N, N1)
            iiR = iiL.redirect()
        A = cytnx.UniTensor ([iiL,ii,iiR], labels=['l','i','r'])
        cytnx.random.uniform_(A, low=-1., high=1.)
        res.append(A)
    norm = mpsut.inner(res,res)
    res[0] *= 1./norm**0.5
    return res

def make_swap (ii1, ii2):
    assert ii1.dim() == ii2.dim() == 2
    swap = cytnx.UniTensor ([ii1,ii2,ii1.redirect(),ii2.redirect()], labels=['ip1','ip2','i1','i2'])
    swap.at([0,0,0,0]).value = 1.
    swap.at([0,1,1,0]).value = 1.
    swap.at([1,0,0,1]).value = 1.
    swap.at([1,1,1,1]).value = 1.
    return swap

# Apply U and Udag directly on the MPO
# U is on site, site+1
def apply_U_mpo (mpo, U, site):
    assert set(U.labels()) == set(['ip1','ip2','i1','i2'])
    UU = U.relabels(['i1','i2'],['_i1up','_i2up'])
    Udag = U.Dagger().relabels(['ip1','ip2','i1','i2'],['i1','i2','_i1dn','_i2dn'])
    A1 = mpo[site].relabels(['i','ip','r'],['_i1dn','_i1up','_'])
    A2 = mpo[site+1].relabels(['i','ip','l'],['_i2dn','_i2up','_'])
    AA = cytnx.Contract(A1,A2)
    AA = cytnx.Contract(AA,UU)
    AA = cytnx.Contract(AA,Udag)
    AA.permute_(['l','ip1','i1','ip2','i2','r'])
    A1, A2 = utut.decompose_tensor (AA, 3, cutoff=1e-14)
    A1.relabels_(['i1','ip1','aux'],['i','ip','r'])
    A2.relabels_(['i2','ip2','aux'],['i','ip','l'])
    mpo[site] = A1
    mpo[site+1] = A2

# Apply a swap gate on i and i+1
def mpo_swap_two_sites_bk (mpo, i):
    ii1 = mpo[i].bond('ip')
    ii2 = mpo[i+1].bond('ip')
    print('i',i)
    swap = make_swap (ii1,ii2)
    apply_U_mpo (mpo, swap, i)
    # swap names
    name = mpo[i].name()
    mpo[i].set_name (mpo[i+1].name())
    mpo[i+1].set_name (name)

# Apply a swap gate on site and site+1
def mpo_swap_two_sites (mpo, site):
    name1, name2 = mpo[site].name(), mpo[site+1].name()
    A1 = mpo[site].relabels(['i','ip','r'],['i1','i1p','_'])
    A2 = mpo[site+1].relabels(['i','ip','l'],['i2','i2p','_'])
    AA = cytnx.Contract(A1,A2)

    AA.permute_(['l','i2p','i2','i1p','i1','r'])
    A1, A2 = utut.decompose_tensor(AA, 3, cutoff=1e-12)
    A1.relabels_(['i2','i2p','aux'],['i','ip','r'])
    A2.relabels_(['i1','i1p','aux'],['i','ip','l'])
    A1.set_name(name2)
    A2.set_name(name1)
    mpo[site] = A1
    mpo[site+1] = A2

# Apply a swap gate on site and site+1
def mps_swap_two_sites (mps, site):
    name1, name2 = mps[site].name(), mps[site+1].name()
    A1 = mps[site].relabels(['i','r'],['i1','_'])
    A2 = mps[site+1].relabels(['i','l'],['i2','_'])
    AA = cytnx.Contract(A1,A2)

    AA.permute_(['l','i2','i1','r'])
    A1, A2 = utut.decompose_tensor(AA, 2, cutoff=1e-12)
    A1.relabels_(['i2','aux'],['i','r'])
    A2.relabels_(['i1','aux'],['i','l'])
    A1.set_name(name2)
    A2.set_name(name1)
    mps[site] = A1
    mps[site+1] = A2

# Here the H is after duplications of sites
def swap_mpo_mps (H, mps, N1):
    N = len(H)
    for i in range(N1,N-N1,2):
        for i1 in range(i-1,i-N1,-1):
            mpo_swap_two_sites (H, i1)
            mps_swap_two_sites (mps, i1)

    for i,di in zip(range(N-N1,N),range(N1,0,-1)):
        for i1 in range(i-1,i-di,-1):
            mpo_swap_two_sites (H, i1)
            mps_swap_two_sites (mps, i1)

if __name__ == '__main__':
    N1 = 3
    rescale = 0.01

    xmax = rescale * (2**N1-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    H1, L1, R1 = tw.get_H_SHO (N1, rescale)

    H, L, R = H_4particles (H1, L1, R1)
    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-10)

    N = len(H)

    H, L, R = split_physical_indices (H, N1)
    mpsut.check_mpo_bonds(H)

    psi = make_init_state_bk2 (N, N1, 16)


    print(mpsut.inner_mpo (psi, psi, H, L, R))


    for h in H:
        print(h.name(),end=' ')
    print()

    swap_mpo_mps (H, psi, N1)

    for h in H:
        print(h.name(),end=' ')
    print()

    print(mpsut.virtual_dims(H))

    print(mpsut.inner_mpo (psi, psi, H, L, R))


    for i in range(len(psi)):
        cytnx.random.uniform_(psi[i], low=-1., high=1.)
    norm = mpsut.inner(psi,psi)
    psi[0] *= 1./norm**0.5


    for i in H:
        print(i.name(),end=' ')
    print()

    #qnut.add_parity_to_mpo (H, L, R)


    '''iis = mpsut.get_physical_bonds(H)
    print(len(H), len(iis))
    psi = make_init_state (iis, N, N1)
    mpsut.check_mps_bonds (psi)'''

    maxdims = [2]*10 + [4]*10 + [8]*10 + [16]*10# + [32]*30
    cutoff = 1e-12

    mpsut.check_mpo_bonds(H)
    mpsut.check_mps_bonds (psi)
    mpsut.check_mps_mpo_physical_bonds (psi, H)



    # Run dmrg
    psi, en = dmrg.dmrg (psi, H, L, R, maxdims, cutoff)



