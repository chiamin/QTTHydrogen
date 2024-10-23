#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt



def inds_to_x (inds, factor=1.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * factor * 2**i
    return res


def make_tensorL (n, factor=1.):
    AL = np.zeros ((2,2))  # (i, k)
    x = factor * 2**n
    # k == 0
    AL[0,0] = 0
    AL[1,0] = np.sin(x)
    # k == 1
    AL[0,1] = 1
    AL[1,1] = np.cos(x)
    return AL

def make_tensorR (n, factor=1.):
    AR = np.zeros ((2,2))  # (k, i)
    x = factor * 2**n
    # k == 0
    AR[0,0] = 1
    AR[0,1] = np.cos(x)
    # k == 1
    AR[1,0] = 0
    AR[1,1] = np.sin(x)
    return AR

def make_LR_sin ():
    L = np.array([0.,1.])
    R = np.array([1.,0.])
    return L, R

def make_LR_cos ():
    L = np.array([1.,0.])
    R = np.array([1.,0.])
    return L, R

def make_tensorA (n, factor=1.):
    A = np.zeros ((2,2,2)) # (k1,i,k2)
    x = factor * 2**n
    # k1 == 0, k2 == 0
    A[0,0,0] = 1
    A[0,1,0] = np.cos(x)
    # k1 == 1, k2 == 0
    A[1,0,0] = 0
    A[1,1,0] = np.sin(x)
    # k1 == 0, k2 == 1
    A[0,0,1] = 0
    A[0,1,1] = -np.sin(x)
    # k1 == 1, k2 == 1
    A[1,0,1] = 1
    A[1,1,1] = np.cos(x)
    return A


def make_sin_qtt (N, factor):
    qtt = []        # Quantic tensor train
    for n in range(N):
        if n == 0:
            A = make_tensorL(n, factor)
        elif n == N-1:
            A = make_tensorR(n, factor)
        else:
            A = make_tensorA(n, factor)
        qtt.append(A)
    return qtt

def make_sin_cos_op (N, rescale):
    qtt = []        # Quantic tensor train
    for n in range(N):
        A = make_tensorA(n, rescale)
        qtt.append(A)
    return qtt


def get_ele (qtt, inds):
    # Reduce to matrix multiplication
    Ms = []                 # Store the matrices after index collapsing
    N = len(qtt)            # The number of tensors in QTT
    for n in range(N):      # For each tensor
        ind = inds[n]       # The index number we want to collapse

        M = qtt[n]
        if n == 0:          # The very left tensor
            M = M[ind,:]
        elif n == N-1:      # The very right tensor
            M = M[:,ind]
        else:               # The central tensors
            M = M[:,ind,:]
        Ms.append(M)
        #print(n,'M =\n',M)

def get_ele(qtt, inds):
    res = np.eye(2)

    N = len(qtt)

    for n in range(N):
        M = qtt[n]

        if n == 0:
            M = M[inds[n], :]
        elif n == N - 1:
            M = M[:, inds[n]]
        else:
            M = M[:, inds[n], :]
        res = np.dot(res, M)

    return res

if __name__ == '__main__':
    N = 10
    factor = 0.01
    qtt = make_sin_qtt (N, factor)

    inds = []
    xs,fs = [],[]
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
                    x = inds_to_x (inds)
                    a = get_ele (qtt, inds)
                    print(inds, x, a)
                    #print('-----------------------------------')
                    xs.append(x)
                    fs.append(a)
    plt.plot (xs, fs)
    #plt.plot (xs, np.sin(xs))
    plt.show()





