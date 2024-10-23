import sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np

A = np.ones((2,2))
U,S,V = np.linalg.svd(A)
print(U)
