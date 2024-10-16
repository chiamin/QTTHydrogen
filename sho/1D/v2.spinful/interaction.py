import random, copy
import plot_utility as ptut
import numpy as np
from ncon import ncon
import npmps, linear
import qtt_utility as ut
import matplotlib.pyplot as plt

def positive_binary_subtract_MPO_tensor ():
    A = np.zeros((2,2,2,2,2))   # (left, b1, b2, b1-b2, right)

    # x1 = (b1_1, b1_2, b1_3, ...)
    # x2 = (b2_1, b2_2, b2_3, ...)
    # Here x = b_1 + b_2 * 2 + b_3 * 4 + ...
    #
    # (left == 0) for site i means that the number (b1_1, b1_2, ..., b1_i-1) < (b2_1, b2_2, ..., b2_i-1)
    # (left == 1) for site i means that the number (b1_1, b1_2, ..., b1_i-1) >= (b2_1, b2_2, ..., b2_i-1)

    # (left == 0): x1 < x2
    #
    # - 0 0  =>  1 -
    # Example: 001-100 = 110
    #
    # - 0 1  =>  0 -
    # Example: 001-110 = 100
    #
    # - 1 0  =>  0 +
    # Example: 011-100 = 101
    #
    # - 1 1  =>  1 -
    # Example: 011-110 = 110
    #
    A[0,0,0,1,0] = 1.
    A[0,0,1,0,0] = 1.
    A[0,1,0,0,1] = 1.
    A[0,1,1,1,0] = 1.
    # (left == 1): x1 >= x2
    #
    # + 0 0  =>  0 +
    # Example: 101-100 = 001
    #
    # + 0 1  =>  1 -
    # Example: 101-110 = 010
    #
    # + 1 0  =>  1 +
    # Example: 111-100 = 011
    #
    # + 1 1  =>  0 +
    # Example: 111-110 = 001
    #
    A[1,0,0,0,1] = 1.
    A[1,0,1,1,0] = 1.
    A[1,1,0,1,1] = 1.
    A[1,1,1,0,1] = 1.
    return A

def negative_binary_subtract_MPO_tensor ():
    A = np.zeros((2,2,2,2,2))   # (left, b1, b2, b1-b2, right)

    A[0,0,0,1,0] = 1.
    A[0,1,0,0,0] = 1.
    A[0,0,1,0,1] = 1.
    A[0,1,1,1,0] = 1.

    A[1,0,0,0,1] = 1.
    A[1,1,0,1,0] = 1.
    A[1,0,1,1,1] = 1.
    A[1,1,1,0,1] = 1.
    return A

def identity_binary_subtract_MPO_tensor():
    A = np.zeros((1,2,2,2,1))   # (left, b1, b2, b1-b2, right)

    A[0,0,0,0,0] = 1.
    A[0,1,1,0,0] = 1.
    return A

def pos_binary_subtract_MPO (N):
    mpo = [positive_binary_subtract_MPO_tensor() for i in range(N)]
    L = np.array([0., 1.])
    R = np.array([0., 1.])
    return mpo, L, R

def neg_binary_subtract_MPO (N):
    mpo = [negative_binary_subtract_MPO_tensor() for i in range(N)]
    L = np.array([0., 1.])
    R = np.array([0., 1.])
    return mpo, L, R

def identity_binary_subtract_MPO (N):
    mpo = [identity_binary_subtract_MPO_tensor() for i in range(N)]
    L = np.array([1.])
    R = np.array([1.])
    return mpo, L, R

def abs_binary_subtract_MPO (N):
    mpo1, L1, R1 = pos_binary_subtract_MPO(N)
    mpo2, L2, R2 = neg_binary_subtract_MPO(N)
    mpoI, LI, RI = identity_binary_subtract_MPO(N)

    # mpo1 + mpo2 - mpoI
    mpo = []
    for n in range(N):
        T1, T2, T3 = mpo1[n], mpo2[n], mpoI[n]
        A = np.zeros((T1.shape[0]+T2.shape[0]+T3.shape[0], T1.shape[1], T1.shape[2], T1.shape[3], T1.shape[4]+T2.shape[4]+T3.shape[4]))
        A[:T1.shape[0],:,:,:,:T1.shape[3]] = T1
        A[T1.shape[0]:T1.shape[0]+T2.shape[0],:,:,:,T1.shape[3]:T1.shape[3]+T2.shape[3]] = T2
        A[T1.shape[0]+T2.shape[0]:,:,:,:,T1.shape[3]+T2.shape[3]:] = T3
        mpo.append(A)

    L = np.concatenate ((L1, L2, -LI))
    R = np.concatenate ((R1, R2, RI))
    return mpo, L, R

def apply_subtract_mpo (mpo, L, R, b1, b2):
    mpo = copy.copy(mpo)
    N = len(mpo)
    assert len(mpo) == len(b1) == len(b2)
    for i in range(N):
        mpo[i] = mpo[i][:,int(b1[i]),int(b2[i]),:,:]
    D = mpo[0].shape[-1]
    mpo[0] = ncon([mpo[0],L], [[1,-1,-2],[1,]]).reshape((1,2,D))
    mpo[-1] = ncon([mpo[-1],R], [[-1,-2,1],[1,]]).reshape((D,2,1))

    re = set()
    for i in range(2**N):
        bstr = ptut.dec_to_bin (i, N)[::-1]
        ele = ptut.get_ele_mps (mpo, bstr)
        if abs(ele) > 1e-12:
            re.add((bstr,ele))
    return re

def check_binary_substracion ():
    N = 4
    mpo1, L1, R1 = pos_binary_subtract_MPO (N)
    mpo2, L2, R2 = neg_binary_subtract_MPO (N)
    mpo, L, R = abs_binary_subtract_MPO (N)

    # Check for one time
    MAX = 2**N-1
    x1 = random.randint(0,MAX)
    x2 = random.randint(0,MAX)
    dx = x1-x2
    bdx = ptut.dec_to_bin (abs(dx), N)[::-1]

    b1 = ptut.dec_to_bin (x1, N)[::-1]
    b2 = ptut.dec_to_bin (x2, N)[::-1]

    print(x1,x2,dx)
    print(b1,b2,bdx)

    res1 = apply_subtract_mpo (mpo1, L1, R1, b1, b2)
    res2 = apply_subtract_mpo (mpo2, L2, R2, b1, b2)
    res = apply_subtract_mpo (mpo, L, R, b1, b2)
    print(res1, res2, res)

    # Check for random binaries
    checkN = 1000
    for i in range(checkN):
        x1 = random.randint(0,MAX)
        x2 = random.randint(0,MAX)
        dx = x1-x2
        bdx = ptut.dec_to_bin (abs(dx), N)[::-1]

        b1 = ptut.dec_to_bin (x1, N)[::-1]
        b2 = ptut.dec_to_bin (x2, N)[::-1]

        res = apply_subtract_mpo (mpo, L, R, b1, b2)
        assert len(res) == 1
        assert list(res)[0][0] == bdx

def abs_subtract_MPS (N):
    # Create x MPS
    xmps = linear.make_x_mps (N)
    # Create binary subtraction MPO
    mpo, L, R = abs_binary_subtract_MPO (N)

    # Apply x MPS to binary subtraction MPO
    #
    #  xMPS  -2 -----o----- -6
    #               _|_               ==>           ___
    #  MPO   -1 ---(___)--- -5                0 ---(___)--- 3
    #              |   |                           |   |
    #             -3  -4                           1   2
    dxmps = []
    for i in range(len(mpo)):
        D1, D2 = xmps[i].shape, mpo[i].shape
        A = ncon([mpo[i],xmps[i]], [[-1,-3,-4,1,-5],[-2,1,-6]])
        A = A.reshape((D1[0]*D2[0], D2[1], D2[2], D1[2]*D2[4]))
        dxmps.append(A)

    # Absort L and R
    L = L.reshape((1,L.shape[0]))
    R = R.reshape((R.shape[0],1))
    dxmps[0] = ncon ([L,dxmps[0]], ((-1,1,), (1,-2,-3,-4)))
    dxmps[-1] = ncon ([R,dxmps[-1]], ((1,-4), (-1,-2,-3,1)))

    return dxmps

def abs_subtract_MPO (N):
    # Create MPS for |x1-x2|
    dxmps = abs_subtract_MPS (N)

    # Reshape the tensors
    #        ___                      ___
    #  0 ---(___)--- 3    ==>   0 ---(___)--- 2
    #       |   |                      |
    #       1   2                      1
    for i in range(len(dxmps)):
        assert dxmps[i].ndim == 4
        D = dxmps[i].shape
        dxmps[i] = dxmps[i].reshape((D[0],D[1]*D[2],D[3]))

    # Make MPS to an MPO
    dxmpo, L, R = npmps.mps_to_mpo (dxmps)

    # Reshape the tensors back
    #
    #       3   4                      2
    #       |___|                     _|_
    #  0 ---(___)--- 5    <==   0 ---(___)--- 3
    #       |   |                      |
    #       1   2                      1
    for i in range(len(dxmps)):
        D = dxmpo[i].shape
        dxmpo[i] = dxmpo[i].reshape((D[0],2,2,2,2,D[-1]))

    return dxmpo, L, R

def apply_abs_subtract_MPS (dxmps, bstr1, bstr2):
    assert len(dxmps) == len(bstr1) == len(bstr2)

    res = dxmps[0][:,int(bstr1[0]),int(bstr2[0]),:]
    for i in range(1,len(dxmps)):
        res = res @ dxmps[i][:,int(bstr1[i]),int(bstr2[i]),:]
    return float(res)

def apply_abs_subtract_MPO (dxmpo, L, R, bstr1, bstr2):
    assert len(dxmpo) == len(bstr1) == len(bstr2)

    # Absort L and R
    L = copy.copy(L)
    R = copy.copy(R)
    L = L.reshape((1,L.shape[0]))
    R = R.reshape((R.shape[0],1))

    res = L
    for i in range(len(dxmpo)):
        res = res @ dxmpo[i][:,int(bstr1[i]),int(bstr2[i]),int(bstr1[i]),int(bstr2[i]),:]
    res = res @ R
    return float(res)

def check_abs_subtract_MPS ():
    N = 4
    dxmpo, L, R = abs_subtract_MPO (N)

    bstr1 = '0001'
    xs = range(2**N)
    ys = []
    for i in xs:
        bstr2 = ptut.dec_to_bin (i, N)[::-1]
        ys.append (apply_abs_subtract_MPO (dxmpo, L, R, bstr1, bstr2))
    plt.plot (xs, ys, marker='.')
    plt.show()

if __name__ == '__main__':
    N = 4

    check_abs_subtract_MPS()
