import os, sys 
#root = os.getenv('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
#sys.path.insert(0,root)
sys.path.append('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
import xfacpy
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
import time
import matplotlib.pyplot as plt

def cc_inds_to_x (inds, rescale=1., shift=0.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift

# The target function
def funQ1 (inds):
    x = cc_inds_to_x (inds)
    x2 = x - 500
    print(inds,x,x2)
    if np.abs(x2) > 1e-10:
        return 1./np.abs(x2)
    else:
        return 1e1

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

#def inds_to_x (inds, x0, dx):
#  s0 = sum([b<<i for i, b in enumerate(inds[::+1])])
#  return x0 + dx*s0

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

def plotF(mps):
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

  #lx = np.linspace(x0, x1, 50000)
  #ly = lx**(-1)
  #plt.plot(lx, ly, 'k-')

  for it in range(2**nsite):
    inds = num_to_inds(it, nsite)
    ff = eval_mps(mps,inds)
    ff2 = funQ1(inds)
    print(inds,ff,ff2)
    xx = cc_inds_to_x (inds)#inds_to_x(inds, x0, dx) 
    plt.plot(xx,ff, c='r', marker='+', ls='None', markersize=4)
    plt.plot(xx,ff2, c='k', marker='x', ls='None', markersize=4)

  #plt.axis([x0,x1, 1E-5, +1E5])
  #plt.yscale('log')

  plt.show()
  plt.tight_layout()
  plt.savefig('tci_inv.pdf', dpi = 600, transparent=True)
  plt.close()

def TCIfit (func, nsite):
  #nsite = 10
  dimP = 2      # Physical dimension
  
  # Fitting range
  #x0 = 1E-3
  #x1 = 1E+3
  #dx = (x1-x0)/(2**nsite)

  minD = 1
  incD = 1      # increasing bond dimension
  maxD = 50

  # Task: find element inverse of [mps0] as [mps]
  mps0 = mps_x1(x0, x1, nsite)

  pm = xfacpy.TensorCI2Param()
  pm.pivot1 = np.random.randint(2, size=nsite)
  pm.reltol = 1e-20
  pm.bondDim = minD
  tci = xfacpy.TensorCI2(funQ1, [dimP]*nsite, pm)

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
  return mps

if __name__ == '__main__':
  nsite = 10
  mps = TCIfit(funQ1, nsite)
  plotF(mps)
