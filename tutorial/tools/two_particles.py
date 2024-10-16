import copy, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
sys.path.append('/home/chiamin/project/2023/qtt/code/new/tools/')
import cytnx
import numpy as np
from ncon import ncon
import npmps
import unitenMPS as uniten
import dmrg as dmrg

def phys_index ():
    return cytnx.Bond(cytnx.BD_IN, [[0],[1]], [3,1], [cytnx.Symmetry.Zn(2)])

# ii is the physical bond
def make_product_mps (N):
    # The parity for the physical bond is [0,0,0,1]
    ii = phys_index()
    # Virtual bonds
    vb0 = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
    vb1 = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])

    # Other sites
    re = []
    for i in range(N):
        if i == 0:
            A = cytnx.UniTensor ([vb0, ii, vb1.redirect()], labels=['l','i','r'])
            A.at([0,3,0]).value = 1.
        else:
            A = cytnx.UniTensor ([vb1, ii, vb1.redirect()], labels=['l','i','r'])
            A.at([0,0,0]).value = 1./3**0.5
            A.at([0,1,0]).value = 1./3**0.5
            A.at([0,2,0]).value = 1./3**0.5
        re.append(A)
    return re

def add_mps_sites_1D (psi, n):
    # Even-parity tensor
    ii = psi[0].bond('i')
    vi = psi[0].bond('l')
    A = cytnx.UniTensor ([vi, ii, vi.redirect()], labels=['l','i','r'])
    A.at([0,0,0]).value = 1./3**0.5
    A.at([0,1,0]).value = 1./3**0.5
    A.at([0,2,0]).value = 1./3**0.5

    phi = [A for i in range(n)]
    return phi + psi

# The parities are 0,0,0,1 for the states after rotation
def get_swap_U ():
    a = 1/2**0.5
    res = np.zeros((2,2,2,2))
    res[0,0,0,0] = 1.
    res[0,1,0,1] = a
    res[0,1,1,0] = a
    res[1,0,0,1] = a
    res[1,0,1,0] = -a
    res[1,1,1,1] = 1.
    res = res.reshape((4,4))

    # Exchange the 3rd and the 4th columns, so that the +1 paritiy are grouped together
    tmp = copy.copy(res[:,2])
    res[:,2] = res[:,3]
    res[:,3] = tmp
    return res

def corr_matrix (psi):
    for j in psi:
        print(j.shape)

    P = np.zeros((2,2,2,2))
    P[0,0,0,0] = 1.
    P[0,1,0,1] = 1.
    P[1,0,1,0] = -1.
    P[1,1,1,1] = 1.
    P = P.reshape((4,4))

    # (I-P)|psi>
    mps = []
    for i in range(len(psi)):
        d = psi[i].shape
        mps.append (psi[i].reshape((d[0],2,2,d[2])))

    mpo = []
    for i in range(len(mps)):
        A = ncon([mps[i],mps[i]], ((-1,-3,1,-5), (-2,-4,1,-6)))
        d = A.shape
        A = A.reshape((d[0]*d[1], d[2], d[3], d[4]*d[5]))
        mpo.append (A)
    return mpo

# mpo is a list of tensors as np.array
def set_mpo_quantum_number (mpo, L, R):
    # Physical bond
    ii = phys_index().redirect()
    iip = ii.redirect()
    # Left virtual bond
    li = cytnx.Bond(cytnx.BD_IN, [[0]], [mpo[0].shape[0]], [cytnx.Symmetry.Zn(2)])
    ri = cytnx.Bond(cytnx.BD_OUT, [[0]], [mpo[-1].shape[-1]], [cytnx.Symmetry.Zn(2)])

    # Set the L and R to UniTensors
    assert L.ndim == 1
    assert R.ndim == 1
    uL = uniten.toUniTen (L.reshape(1,L.shape[0],1))
    uR = uniten.toUniTen (R.reshape(1,R.shape[0],1))
    vb0_in = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
    vb1_in = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])
    vb0_out = vb0_in.redirect()
    vb1_out = vb1_in.redirect()
    qnL = cytnx.UniTensor ([li.redirect(), vb0_out, vb0_in], labels=['mid','dn','up'])
    qnR = cytnx.UniTensor ([ri.redirect(), vb1_in, vb1_out], labels=['mid','dn','up'])
    qnL.convert_from(uL)
    qnR.convert_from(uR)

    # For checking
    mpo0 = copy.copy(mpo)
    mpo = copy.copy(mpo)

    re = []
    # mpoA is the tensor we want to give quantum numbers
    mpoA = mpo[0]
    for i in range(len(mpo)-1):
        # right dimension
        rdim = mpoA.shape[3]
        # right bond for each quantum number sector
        ri0 = cytnx.Bond(cytnx.BD_OUT, [[0]], [rdim], [cytnx.Symmetry.Zn(2)])
        ri1 = cytnx.Bond(cytnx.BD_OUT, [[1]], [rdim], [cytnx.Symmetry.Zn(2)])

        # The bonds "l", "ip" and "i" have well-defined parities, but the bond "r" does not.
        # To define parity to the bond "r", we first 
        # Get Tensor for each quantum number sector
        # qn_T0 is the sub-tensor that has 0 parity on the ri0 bond
        # qn_T1 is the sub-tensor that has 1 parity on the ri1 bond
        uT = uniten.toUniTen(mpoA)
        qn_T0 = cytnx.UniTensor([li,iip,ii,ri0], labels=["l","ip","i","r"])
        qn_T1 = cytnx.UniTensor([li,iip,ii,ri1], labels=["l","ip","i","r"])
        qn_T0.convert_from(uT, force=True)
        qn_T1.convert_from(uT, force=True)
        qn_Ts = [qn_T0, qn_T1]

        # Expand the "r" bond to contain 0 and 1 pairty sectors
        # direct sum of qn_T0 and qn_T1
        ri = cytnx.Bond(cytnx.BD_OUT, [[0],[1]], [rdim, rdim], [cytnx.Symmetry.Zn(2)])
        qn_T = cytnx.UniTensor([li,iip,ii,ri], labels=["l","ip","i","r"])
        for i1 in range(len(qn_T.bond("l").qnums())):
          for i2 in range(len(qn_T.bond("ip").qnums())):
            for i3 in range(len(qn_T.bond("i").qnums())):
              for i4 in range(len(qn_T.bond("r").qnums())):
                # i4 specifies the parity of the "r" bond
                # Get the 0 or 1 parity block from qn_Ts (qn_T0 or qn_T1)
                blk = qn_Ts[i4].get_block(["l","ip","i","r"],[i1,i2,i3,0], force=True)
                if blk.rank() != 0:
                    # Put the block to the enlarged tensor qn_T
                    qn_T.put_block_(blk, ["l","ip","i","r"],[i1,i2,i3,i4])

        # mpoA2 is the MPO tensor for the next site
        # Expand the "l" bond for mpoA2
        # so that the original mpo[i]*mpo[i+1] is equal to qn_T * mpoA2
        rrdim = mpo[i+1].shape[3]
        mpoA2 = np.zeros((2*rdim,4,4,rrdim))   # Expand "l" from rdim to 2*rdim
        mpoA2[:rdim,:,:,:] = mpo[i+1]
        mpoA2[rdim:,:,:,:] = mpo[i+1]

        # SVD qn_T = A*s*vt
        # A is the symmetric MPO tensor we want
        qn_T.set_rowrank_(3)
        s, A, vt = cytnx.linalg.Svd_truncate(qn_T, keepdim=2*rdim, err=1e-12)
        A.relabel_("_aux_L","r")
        re.append(A)

        # R = s*vt
        # Now the original mpo[i]*mpo[i+1] = A * (R * mpoA2)
        R = cytnx.Contract(s,vt)

        # Convert R to np.array
        TR = cytnx.UniTensor.zeros(R.shape())
        TR = TR.convert_from(R).get_block().numpy()
        # Obsort R to mpoA2
        # Now the original mpo[i]*mpo[i+1] = A * mpoA2
        # Note that mpoA2 is a np.array
        mpoA2 = ncon([TR, mpoA2], ((-1,1), (1,-4,-5,-6)))

        # Set mpoA for the next site
        mpoA = mpoA2
        # Set the left bond
        li = A.bond("r").redirect()

    # Get the MPO tensor for the last site
    # Convert mpoA2 to a non-symmetric uniTensor
    uT = uniten.toUniTen (mpoA2)
    # Convert uT to a symmetric uniTensor
    ri = cytnx.Bond(cytnx.BD_OUT, [[0]], [mpo[-1].shape[-1]], [cytnx.Symmetry.Zn(2)])
    qn_T = cytnx.UniTensor([li,iip,ii,ri], labels=["l","ip","i","r"])
    qn_T.convert_from (uT, force=True)
    re.append(qn_T)

    #check_mpo_bond_directions (re)
    return re, qnL, qnR

def one_particle_denmat (psi, U, maxdims, cutoff):
    psi = uniten.mps_to_nparray (psi)
    psi = npmps.applyLocal_mps (psi, U)

    corr = corr_matrix (psi)
    Lcorr = np.array([1.])
    Rcorr = np.array([1.])

    for i in corr:
        print(i.shape)

    # Target the largest occupations
    corr[0] *= -1.
    corr, Lcorr, Rcorr = npmps.compress_mpo (corr, Lcorr, Rcorr, cutoff=1e-12)
    corr, Lcorr, Rcorr = uniten.npmpo_to_uniten (corr, Lcorr, Rcorr)

    # First particle
    N = len(psi)
    phi = npmps.random_MPS (N, phydim=2, vdim=2)
    phi = uniten.npmps_to_uniten (phi)
    phi, occs, terrs = dmrg.dmrg (phi, corr, Lcorr, Rcorr, maxdims, cutoff, maxIter=4)

    # Second particle
    phi1 = npmps.random_MPS (N, phydim=2, vdim=2)
    phi1 = uniten.npmps_to_uniten (phi1)
    phi1, occs1, terrs = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff, maxIter=4, ortho_mpss=[phi], weights=[20])

    npphi1 = uniten.to_npMPS (phi)
    npphi2 = uniten.to_npMPS (phi1)
    return npphi1, npphi2
