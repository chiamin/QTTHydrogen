import qtt_utility as ut
import copy
import numpy as np
import npmps

def dec_to_bin (dec, N):
    bstr = ("{:0>"+str(N)+"b}").format(dec)
    return bstr

def bin_to_dec (bstr, rescale=1., shift=0.):
    assert type(bstr) == str
    return int(bstr[::-1],2) * rescale + shift

def bin_to_dec_list (bstrs, rescale=1., shift=0.):
    return np.array([bin_to_dec (bstr, rescale, shift) for bstr in bstrs])

# An iterator for binary numbers
class BinaryNumbers:
    # N is the number of "site", or the number of binary numbers
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N     # The largest decimal number + 1

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin (dec, self.N_num)[::-1]
        else:
            raise StopIteration

def get_ele_mps (mps, bstr):
    assert type(bstr) == str
    assert len(mps) == len(bstr)
    # Reduce to matrix multiplication
    res = [[1.]]
    for i in range(len(mps)):
        A = mps[i]
        bi = bstr[i]

        M = A[:,int(bi),:]
        res = res @ M
    return float(res)

def get_ele_mpo (mpo, L, R, bstr):
    assert type(bstr) == str
    assert len(mpo) == len(bstr)

    # Absort L and R into mpo
    mpo = npmps.absort_LR (mpo, L, R)

    # Reduce to matrix multiplication
    res = [[1.]]
    for i in range(len(mpo)):
        A = mpo[i]
        bi = int(bstr[i])

        M = A[:,bi,bi,:]
        res = res @ M
    return float(res)

# Return a ufunc that returns the element of a 2D function in the QTT format
# A ufunc can automatically apply on all the input elements
# bx and by are binary-number strings for x and y
#
# The returned function can be used as follows:
# uf = ufunc_2D_eles (mps)
# res = uf (bxs, bys)
# where bxs and bys can be lists of binary-number strings
#
def ufunc_2D_eles_mps (mps):
    # Define a function that only x and y are input parameters
    def _get_ele (bx, by):
        # bx + by combine the binary strings to a single string
        return get_ele_mps (mps, bx+by)

    # Return a ufunc
    return np.frompyfunc (_get_ele, 2, 1)

def get_2D_mesh_eles_mps (mps, bxs, bys):
    bX, bY = np.meshgrid (bxs, bys)
    get_2D_ele = ufunc_2D_eles_mps (mps)  # ufunc
    fs = get_2D_ele (bX, bY)
    fs = fs.astype(np.float64)
    return fs

def get_2D_mesh_eles_mpo (mpo, L, R, bxs, bys):

    # Define a function that only x and y are input parameters
    def _get_ele (bx, by):
        # bx + by combine the binary strings to a single string
        return get_ele_mpo (mpo, L, R, bx+by)

    # Make a ufunc
    get_2D_ele = np.frompyfunc (_get_ele, 2, 1)

    bX, bY = np.meshgrid (bxs, bys)
    fs = get_2D_ele (bX, bY)
    fs = fs.astype(np.float64)
    return fs


if __name__ == '__main__':
    print(list(BinaryNumbers(4)))
    exit()

    a = [[1,2],[3,4]]
    b = [[10,20],[30,40]]
    def myf (x,y): return x+y
    myff = np.frompyfunc(myf,2,1)
    print (myff (a,b))
    exit()

    # An example of using BinaryNumbers
    for x in BinaryNumbers(4):
        print(x)
