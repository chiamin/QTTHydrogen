import copy, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
import cytnx_dmrg as dmrg

def toUniTen (T):
    assert type(T) == np.ndarray
    T = cytnx.from_numpy(T)
    return cytnx.UniTensor (T)

def to_nparray(T):
    assert type(T) == cytnx.UniTensor
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

def check_same_bonds (b1, b2):
    assert b1.type() == b2.redirect().type()
    assert b1.qnums() == b2.qnums()
    assert b1.getDegeneracies() == b2.getDegeneracies()

def decompose_tensor (T, rowrank, leftU=True, dim=sys.maxsize, cutoff=0.):
    # 1. SVD
    T.set_rowrank_(rowrank)
    s, A1, A2 = cytnx.linalg.Svd_truncate (T, keepdim=dim, err=cutoff)
    # 2. Absort s to A2 or A1
    if leftU:
        A2 = cytnx.Contract(s,A2)
        A1.relabel_('_aux_L','aux')
        A2.relabel_('_aux_L','aux')
    else:
        A1 = cytnx.Contract(s,A1)
        A1.relabel_('_aux_R','aux')
        A2.relabel_('_aux_R','aux')
    return A1, A2

def print_bond (bond):
    print(bond.type(), bond.qnums(), bond.getDegeneracies())

def print_bonds (T):
    print(T.labels())
    print(T.shape())
    for i in T.bonds():
        print_bond(i)

# ============================== Convert between numpy array and UniTensor ==============================
def np_to_MPS_tensor (T):
    A = cytnx.UniTensor (cytnx.from_numpy(T), rowrank=2)
    A.set_labels(['l','i','r'])
    return A

def npmps_to_uniten (mps):
    res = []
    for i in range(len(mps)):
        A = np_to_MPS_tensor(mps[i])
        res.append(A)
    return res

def npmpo_to_uniten (mpo,L,R):
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

def to_npMPS (mps):
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

# ============================================================

# bx is x in binary format
def get_ele (mps, bx):
    res = mps[0].get_block().numpy()[:,int(bx[0]),:]
    for bi,A in zip(bstr,mps):
        M = A.get_block().numpy()
        M = A[:,int(bi),:]
        res = res @ M
    return float(res)

# mpo is a list of UniTensors
def check_mpo_bonds (mpo, L, R):
    assert set(L.labels()) == set(['mid','up','dn'])
    assert set(R.labels()) == set(['mid','up','dn'])
    assert L.bond('up').dim() == L.bond('dn').dim() == 1
    assert R.bond('up').dim() == R.bond('dn').dim() == 1
    check_same_bonds (L.bond('mid'), mpo[0].bond('l'))
    check_same_bonds (R.bond('mid'), mpo[-1].bond('r'))
    for i in range(len(mpo)):
        assert set(mpo[i].labels()) == set(['l','ip','i','r'])
        ii = mpo[i].bond('i')
        iip = mpo[i].bond('ip')
        check_same_bonds(ii,iip)

        if i != len(mpo)-1:
            b1 = mpo[i].bond("r")
            b2 = mpo[i+1].bond("l")
            check_same_bonds(b1,b2)

def check_mps_bonds (mps):
    for i in range(len(mps)):
        assert set(mps[i].labels()) == set(['l','i','r'])

        if i != len(mps)-1:
            b1 = mps[i].bond("r")
            b2 = mps[i+1].bond("l")
            check_same_bonds(b1,b2)
    assert mps[0].bond('l').dim() == mps[-1].bond('r').dim() == 1

def check_mps_mpo_physical_bonds (mps, mpo):
    print(len(mps), len(mpo))
    assert len(mps) == len(mpo)
    for i in range(len(mps)):
        ii = mps[i].bond("i")
        ii2 = mpo[i].bond("i")
        check_same_bonds(ii,ii2)

# Get the incoming physical bonds
def get_physical_bonds (mpo):
    return [A.bond("ip") for A in mpo]

def inner (mps1, mps2):
    assert len(mps1) == len(mps2)
    LR = dmrg.LR_envir_tensors_mps (len(mps1), mps1, mps2)
    LR.update_LR (mps1, mps2, -1)
    re = LR[0]
    assert re.Nblocks() == 1
    return re.get_block(0).item()

def inner_mpo (mps1, mps2, mpo, L, R):
    assert len(mps1) == len(mps2) == len(mpo)
    LR = dmrg.LR_envir_tensors_mpo (len(mps1), L, R)
    LR.update_LR (mps1, mps2, mpo, -1)
    re = cytnx.Contract(LR[0], L)
    assert re.Nblocks() == 1
    return re.get_block(0).item()

def virtual_dims (mps):
    dims = [mps[0].bond("l").dim()]
    for A in mps:
        dims.append(A.bond("r").dim())
    return dims

def max_dim (mps):
    dims = virtual_dims (mps)
    return max(dims)

# In-place change the MPO
# Merge and SVD the tensors at sites i and i+1,
# and absorb the sigular-value tensor to the left or the right tensor
# depending on the argument <leftU>
def svd_bond_mpo (mpo, i, leftU, dim=sys.maxsize, cutoff=0.):
    A1 = mpo[i].relabels(['i','ip','r'],['i1','ip1','_'])
    A2 = mpo[i+1].relabels(['i','ip','l'],['i2','ip2','_'])
    AA = cytnx.Contract(A1,A2)
    AA.permute_(['l','ip1','i1','ip2','i2','r'])
    A1, A2 = decompose_tensor (AA, rowrank=3, dim=dim, cutoff=cutoff, leftU=leftU)
    A1.relabels_(['i1','ip1','aux'],['i','ip','r'])
    A2.relabels_(['i2','ip2','aux'],['i','ip','l'])
    mpo[i], mpo[i+1] = A1, A2

def reorthogonalize_mpo (mpo, D=sys.maxsize, cutoff=0.):
    mpo = np.copy(mpo)
    check_mpo_bonds (mpo)

    for i in range(len(mpo)-1):
        svd_bond_mpo (mpo, i, leftU=True)
    for i in range(len(mpo)-2,-1,-1):
        svd_bond_mpo (mpo, i, leftU=False)

    check_mpo_bonds (mpo)
    return mpo

# In-place change the MPS
def svd_bond_mps (mps, i, leftU, dim=sys.maxsize, cutoff=0.):
    A1 = mps[i].relabels(['i','r'],['i1','_'])
    A2 = mps[i+1].relabels(['i','l'],['i2','_'])
    AA = cytnx.Contract(A1,A2)
    AA.permute_(['l','i1','i2','r'])
    A1, A2 = decompose_tensor (AA, rowrank=2, dim=dim, cutoff=cutoff, leftU=leftU)
    A1.relabels_(['i1','aux'],['i','r'])
    A2.relabels_(['i2','aux'],['i','l'])
    mps[i], mps[i+1] = A1, A2

def reorthogonalize_mps (mps, D=sys.maxsize, cutoff=0.):
    mps = np.copy(mps)
    check_mps_bonds (mps)

    for i in range(len(mps)-1):
        svd_bond_mps (mps, i, leftU=True)
    for i in range(len(mps)-2,-1,-1):
        svd_bond_mps (mps, i, leftU=False)

    check_mps_bonds (mps)
    return mps

