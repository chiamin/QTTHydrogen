import sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np

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
