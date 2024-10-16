import os, sys 
#root = os.getenv('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
#sys.path.insert(0,root)
sys.path.append('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
import xfacpy

import numpy as np
import time
import matplotlib.pyplot as plt

def cc_inds_to_x (inds, rescale=1., shift=0.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift

def get_dxdy (inds):
    #print(inds)
    xx1 = inds[::2]
    xx2 = inds[1::2]
    N = len(xx1)//2
    x1 = xx1[:N]
    y1 = xx1[N:]
    x2 = xx2[:N]
    y2 = xx2[N:]
    #print(x1,y1,x2,y2)
    x1 = cc_inds_to_x (x1)
    y1 = cc_inds_to_x (y1)
    x2 = cc_inds_to_x (x2)
    y2 = cc_inds_to_x (y2)
    return x2-x1, y2-y1

def get_dx (inds):
    x1 = inds[::2]
    x2 = inds[1::2]
    x1 = cc_inds_to_x (x1)
    x2 = cc_inds_to_x (x2)
    return abs(x2-x1)

def funQ1 (inds, cutoff):
    dx = get_dx (inds)
    if dx < cutoff:
        dx = cutoff
    return dx**(-1)

def funQ1_2D (inds):
    dx,dy = get_dxdy (inds)
    r = np.sqrt(dx**2 + dy**2)
    if np.abs(r) > 1e-10:
        return r**(-1)
    else:
        return 1e10

def mps_x1(x0, x1, nsite):
  mps = []
  t0 = (x0/nsite)*np.array([1.,1.])
  dx = x1-x0

  for it in range(nsite):
    ten = np.zeros((2,2,2))
    ten[0,:,0] = [1.,1.]
    ten[1,:,1] = [1.,1.]

    fac = dx/(2**(nsite-it))
    tx = t0 + [0., fac]
    ten[1,:,0] = tx
    mps.append(ten)

  mps[0] = mps[0][1:2,:,:]
  mps[-1] = mps[-1][:,:,0:1]
  return mps 

def inds_to_x (inds, x0, dx):
  s0 = sum([b<<i for i, b in enumerate(inds[::+1])])
  return x0 + dx*s0

def num_to_inds(num, nsite):
  inds = [int(it) for it in np.binary_repr(num, width=nsite)]
  return inds[::-1]

def eval_mps (mps, inds, envL=np.array([1.]), envR=np.array([1.])):
  nsite = len(mps)
  val = envL
  for it in range(nsite):
    ind = inds[it]
    mat = mps[it][:,ind,:]
    val = val @ mat 
  val = val @ envR
  return val 

def xfac_to_npmps (mpsX, nsite):
  mps = [None for i in range(nsite)]
  for it in range(nsite):
    mps[it] = mpsX.get(it)

  return mps 

def plotF(mps, target_func):
  plt.rcParams['figure.figsize'] = 6,3 
  SMALL_SIZE = 18
  MEDIUM_SIZE = 20
  BIGGER_SIZE = 16
  plt.rc('font', size=SMALL_SIZE)    
  plt.rc('axes', titlesize=SMALL_SIZE)    
  plt.rc('axes', labelsize=MEDIUM_SIZE)
  plt.rc('xtick', labelsize=SMALL_SIZE)    
  plt.rc('ytick', labelsize=SMALL_SIZE)    
  plt.rc('legend', fontsize=16)    
  plt.rc('figure', titlesize=BIGGER_SIZE)  

  lx = np.linspace(x0, x1, 50000)
  ly = lx**(-1)
  #plt.plot(lx, ly, 'k-')

  for it in range(2**nsite):
    inds = num_to_inds(it, nsite)
    ff = eval_mps(mps,inds)
    ff2 = target_func(inds)
    xx = inds_to_x(inds, x0, dx) 
    plt.plot(xx,ff, c='r', marker='+', ls='None', markersize=4)
    plt.plot(xx,ff2, c='k', marker='x', ls='None', markersize=4)

  #plt.axis([x0,x1, 1E-5, +1E5])
  #plt.yscale('log')
  #plt.ylim(ymin=-100)
  plt.show()
  #plt.tight_layout()
  #plt.savefig('tci_inv.pdf', dpi = 600, transparent=True)
  #plt.close()

def write_mps (fname, mps):
    tmp = np.array(mps, dtype=object)
    np.save(fname,tmp, allow_pickle=True)

def load_mps (fname):
    tmp = np.load(fname, allow_pickle=True)
    return list(tmp)
        

if __name__ == '__main__':
    nsite = int(sys.argv[1])
    dimP = 2      # Physical dimension
    rescale = float(sys.argv[2])
    cutoff = 1e-4#float(sys.argv[3])
    factor = 1#float(sys.argv[4])

    # Fitting range
    x0 = 1E-3
    x1 = 1E+3
    dx = (x1-x0)/(2**nsite)

    minD = 1
    incD = 1      # increasing bond dimension
    maxD = 50

    # Task: find element inverse of [mps0] as [mps]
    mps0 = mps_x1(x0, x1, nsite)

    def target_func (inds):
        return funQ1(inds, cutoff)

    pm = xfacpy.TensorCI2Param()
    pm.pivot1 = np.random.randint(2, size=nsite)
    pm.reltol = 1e-20
    pm.bondDim = minD
    tci = xfacpy.TensorCI2(target_func, [dimP]*nsite, pm)

    while tci.param.bondDim < maxD:
        tci.param.bondDim = tci.param.bondDim + incD

        t0 = time.time()
        tci.iterate(2,2)
        err0 = tci.pivotError[0]
        err1 = tci.pivotError[-1]

        print("{0:20.3e} {1:20.3e} {2:20.3e} {3:20.2e}".
             format(err0, err1, err1/err0, time.time()-t0))

        if (err1/err0 < 1e-13):
            break

    # Output MPS
    mps = xfac_to_npmps (tci.tt, nsite)

    write_mps ('fit.mps', mps)

    plotF(mps, target_func)
