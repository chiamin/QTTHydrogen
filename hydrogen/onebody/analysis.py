import numpy as np
import matplotlib.pyplot as plt
import sys

def trunc_err (fname):
    ms,terrs,ens = np.loadtxt(fname)
    dat = dict()
    for i in range(len(ms)):
        dat[ms[i]] = [terrs[i], ens[i]]

    ms = sorted(list(set(ms)))
    terrs, ens = [],[]
    for m in ms:
        print(m,*dat[m])
        terrs.append(dat[m][0])
        ens.append(dat[m][1])

    plt.plot (terrs, ens, marker='o')
    plt.show()

if __name__ == '__main__':
    fname = sys.argv[1]
    trunc_err(fname)
