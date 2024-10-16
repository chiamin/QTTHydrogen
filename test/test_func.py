import sys, copy
sys.path.append('../tools')
import sincos as sc
import plot_utility as ptut
import matplotlib.pyplot as plt
import numpy as np
import differential as dff
import npmps

if __name__ == '__main__':
    N = 5
    x1,x2 = -np.pi,np.pi

    fig, ax = plt.subplots()

    # sin function
    f = sc.sin_mps (N, x1, x2)
    ptut.plot_1D(f, x1, x2, ax=ax, marker='o')

    # Add one site
    f2 = copy.copy(f)
    f2.insert(0,np.array([1.,1.]).reshape(1,2,1))
    f2.insert(0,np.array([1.,1.]).reshape(1,2,1))
    ptut.plot_1D(f2, x1, x2, ax=ax, marker='o')
    plt.show()

    # Change one tensor
    f2 = copy.copy(f)
    for i in range(4):
        #f2[0] = np.random.uniform(-1.,1.,f2[0].shape)
        f2[0] += np.random.uniform(-0.5,0.5,f2[0].shape)
        f2[1] += np.random.uniform(-0.5,0.5,f2[1].shape)
        f2[2] += np.random.uniform(-0.5,0.5,f2[2].shape)
        #f2[-2] += np.random.uniform(-0.5,0.5,f2[-2].shape)
        #f2[-1] += np.random.uniform(-0.5,0.5,f2[-1].shape)
        ptut.plot_1D(f2, x1, x2, ax=ax, marker='o')
    plt.show()

