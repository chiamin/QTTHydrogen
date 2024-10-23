import numpy as np
import matplotlib.pyplot as plt
import Ex_sin as ss
from ncon import ncon
import qtt_utility as ut

def make_tensorA ():
    A = np.zeros ((3,2,2,3)) # (k1,ipr,i,k2)
    A[0,:,:,0] = ut.I
    A[1,:,:,0] = ut.sp
    A[2,:,:,0] = ut.sm
    A[1,:,:,1] = ut.sm
    A[2,:,:,2] = ut.sp
    return A

def make_LR ():
    L = np.array([-2,1,1])
    R = np.array([1,1,1])
    return L, R

# QTT for a (d/dx)^2 operator
def make_d2dx2_optt (N):
    op_qtt = [make_tensorA() for n in range(N)]        # QTT for a (d/dx)^2 operator
    L = np.array([-2,1,1])
    R = np.array([1,1,1])
    op_qtt[0] = ncon ([L,op_qtt[0]], ((1,), (1,-1,-2,-3)))
    op_qtt[-1] = ncon ([R,op_qtt[-1]], ((1,), (-1,-2,-3,1)))
    return op_qtt

def project_qtt_op (op_qtt, inds):
    res = []
    N = len(op_qtt)
    for n in range(N):
        M = op_qtt[n]
        # Project the operator tensor
        if n == 0:
            M = M[inds[n],:,:]      # (ipr, i, k) --> (i,k)
        elif n == N - 1:
            M = M[:,inds[n],:]      # (k, ipr, i) --> (k,i)
        else:
            M = M[:,inds[n],:,:]    # (k1,ipr,i,k2) --> (k1,i,k2)
        res.append(M)
    return res

def contract_qtt (qtt1, qtt2):
    res = ncon ([qtt1[0],qtt2[0]], ((1,-1), (1,-2)))
    for i in range(1,len(qtt2)-1):
        res = ncon ([res,qtt1[i],qtt2[i]], ((1,2), (1,3,-1), (2,3,-2)))
    res = ncon([res,qtt1[-1],qtt2[-1]], ((1,2), (1,3), (2,3)))
    return res

if __name__ == '__main__':
    # Create a (d/dx)^2 operator
    N = 10             # Number of tensors in the QTT operator
    op_qtt = make_d2dx2_optt(N)

    # Create a sin function
    factor = 0.01
    sin_qtt = ss.make_sin_qtt (N, factor)  # A sine function in QTT form

    # Apply 
    inds = []
    xs,fs,d2fs = [],[],[]
    for i10 in [0,1]:
     for i9 in [0,1]:
      for i8 in [0,1]:
       for i7 in [0,1]:
        for i6 in [0,1]:
         for i5 in [0,1]:
          for i4 in [0,1]:
           for i3 in [0,1]:
            for i2 in [0,1]:
                for i1 in [0,1]:
                    inds = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]
                    x = ut.inds_to_x (inds, factor)


                    # Project qtt operator
                    op_qtt_proj = project_qtt_op (op_qtt, inds)
                    f = ss.get_ele (sin_qtt, inds)
                    d2f = contract_qtt (op_qtt_proj, sin_qtt)

                    # Contract op_qtt_proj and sin_qtt to get the value (a number)
                    # Write your code here
                    

                    xs.append(x)
                    fs.append(f)
                    d2fs.append(d2f)
    d2fs = np.array(d2fs)
    d2fs[0] = d2fs[-1] = None   # First and last points are ill-defined
    d2fs *= 1/factor**2
    plt.plot (xs, fs)
    plt.plot (xs, d2fs)
    plt.show()


