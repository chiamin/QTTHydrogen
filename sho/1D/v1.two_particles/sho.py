import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon

# MPO tensor:
#                 (ipr)                   (ipr)                        (ipr)
#                   1                       0                            1
#                   |                       |                            |
#         (k1)  0 --o-- 3 (k2)              o-- 2 (k)           (k)  0 --o
#                   |                       |                            |
#                   2                       1                            2
#                  (i)                     (i)                          (i)
#
#
#                   2                       0                            2
#                   |                       |                            |
#           T1  0 --o-- 4                   o-- 2                    0 --o
#                   |                       |                            |
#           T2  1 --o-- 5                   o-- 3                    1 --o
#                   |                       |                            |
#                   3                       1                            3
#
#
#                   1                       0                           1
#                   |                       |                           |
#               0 --o-- 2                   o-- 1                   0 --o
#
def prod_mpo_tensor (T1, T2):
    di = T2.shape[2]
    dipr = T1.shape[1]
    dk1 = T1.shape[0] * T2.shape[0]
    dk2 = T1.shape[3] * T2.shape[3]

    T = ncon ([T1,T1], ((-1,-3,1,-5), (-2,1,-4,-6)))
    T = T.reshape ((dk1,dipr,di,dk2))
    return T

def sum_mpo_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3]+T2.shape[3]))
    res[:T1.shape[0],:,:,:T1.shape[3]] = T1
    res[T1.shape[0]:,:,:,T1.shape[3]:] = T2
    return res

def get_H_SHO (N, rescale):
    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        x2_tensor = prod_mpo_tensor (x_tensor, x_tensor)
        ddx2_tensor = df.make_tensorA()
        hi = sum_mpo_tensor (x2_tensor, ddx2_tensor)
        H.append(hi)

    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    L_ddx2, R_ddx2 = df.make_LR()
    L = np.concatenate ((L_x2, -L_ddx2))
    R = np.concatenate ((R_x2, R_ddx2))
    return H, L, R

if __name__ == '__main__':
    N = 12
    rescale = 0.01

    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    xmpo, x2mpo = [], []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        x2_tensor = prod_mpo_tensor (x_tensor, x_tensor)
        ddx2_tensor = df.make_tensorA()
        xmpo.append (x_tensor)
        x2mpo.append (x2_tensor)

    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    L_ddx2, R_ddx2 = df.make_LR()

    H, L, R = get_H_SHO (N, rescale)

    H = lin.contract_LR (H, L, R)
    xmpo = lin.contract_LR (xmpo, L_x, R_x)
    x2mpo = lin.contract_LR (x2mpo, L_x2, R_x2)

    sin_qtt = ss.make_sin_qtt (N, rescale)  # A sine function in QTT form

    inds = []
    xs,fs,f2s,fsins,Hsins = [],[],[],[],[]
    for i12 in [0,1]:
     for i11 in [0,1]:
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
                    inds = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]
                    x = ut.inds_to_x (inds, rescale)
                    xs.append(x)
                    fs.append(ut.get_ele_op (xmpo, inds))
                    f2s.append(ut.get_ele_op (x2mpo, inds))

                    # Project qtt operator
                    fsins.append(ss.get_ele (sin_qtt, inds))
                    H_proj = df.project_qtt_op (H, inds)
                    Hsin = df.contract_qtt (H_proj, sin_qtt)
                    Hsins.append (Hsin)
    fsins = np.array(fsins)
    fs = np.array(fs)

    f2s[-1] = float('Nan')
    plt.plot (xs, fs, label='$x$')
    plt.plot (xs, f2s, label='$x^2$')
    plt.legend()

    plt.figure()
    plt.plot (xs, fsins, label='sin')
    plt.plot (xs, Hsins, label='Hsin')

    gg = fsins + (f2s * fsins)
    plt.plot (xs, gg, label='exact')
    plt.legend()
    plt.show()

