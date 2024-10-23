import numpy as np

def truncate_svd2 (T, rowrank, cutoff):
    ds = T.shape
    d1, d2 = 1, 1
    ds1, ds2 = [],[]
    for i in range(rowrank):
        d1 *= ds[i]
        ds1.append(ds[i])
    for i in range(rowrank,len(ds)):
        d2 *= ds[i]
        ds2.append(ds[i])
    T = T.reshape((d1,d2))
    U, S, Vh = np.linalg.svd (T)
    U = U[:,:len(S)]
    Vh = Vh[:len(S),:]
    ii = (S >= cutoff)
    U, S, Vh = U[:,ii], S[ii], Vh[ii,:]

    A = (U*S).reshape(*ds1,-1)
    B = Vh.reshape(-1,*ds2)
    return A, B

