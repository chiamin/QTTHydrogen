import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import copy, sys
sys.path.append('/home/chiamin/mypy/')
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import qn_utility as qn
import MPS_utility as mpsut
import plot_utility as ptut
import twobody as twut
import npmps
import plotsetting as ps

def get_H_SHO (N, rescale):
    xmax = rescale * (2**N-1)
    shift = -xmax/2
    print('xmax =',xmax)
    print('xshift =',shift)

    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        x2_tensor = ut.prod_mpo_tensor (x_tensor, x_tensor)
        ddx2_tensor = df.make_tensorA()
        hi = ut.sum_mpo_tensor (x2_tensor, ddx2_tensor)
        H.append(hi)

    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    L_ddx2, R_ddx2 = df.make_LR()
    L = np.concatenate ((L_x2, -L_ddx2))
    R = np.concatenate ((R_x2, R_ddx2))
    return H, L, R

def get_ele (qtt, inds):
    res = qtt[0].get_block().numpy()[:,inds[0],:]
    N = len(qtt)
    for n in range(1,N):
        M = qtt[n].get_block().numpy()
        M = M[:,inds[n],:]
        res = np.dot(res, M)
    return float(res)

# The parities are 0,0,0,1 for the states after rotation
def get_swap_U ():
    a = 1/2**0.5
    res = np.zeros((2,2,2,2))
    res[0,0,0,0] = 1.
    res[0,1,0,1] = a
    res[0,1,1,0] = a
    res[1,0,0,1] = a
    res[1,0,1,0] = -a
    res[1,1,1,1] = 1.
    res = res.reshape((4,4))

    # Exchange the 3rd and the 4th columns, so that the +1 paritiy are grouped together
    tmp = np.copy(res[:,2])
    res[:,2] = res[:,3]
    res[:,3] = tmp
    return res

def H_two_particle2 (H1, L1, R1):
    I = np.array([[1.,0.],[0.,1.]])

    N = len(H1)
    H = []
    for i in range(N):
        HIi = ncon([H1[i],I], ((-1,-2,-4,-6), (-3,-5)))
        IHi = ncon([H1[i],I], ((-1,-3,-5,-6), (-2,-4)))
        d = H1[i].shape
        HIi = HIi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        IHi = IHi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        H.append (ut.sum_mpo_tensor (HIi, IHi))

    L = np.append(L1,L1)
    R = np.append(R1,R1)
    return H, L, R

def corr_matrix (psi):
    P = np.zeros((2,2,2,2))
    P[0,0,0,0] = 1.
    P[0,1,0,1] = 1.
    P[1,0,1,0] = -1.
    P[1,1,1,1] = 1.
    P = P.reshape((4,4))

    # (I-P)|psi>
    mps = []
    for i in range(len(psi)):
        d = psi[i].shape
        mps.append (psi[i].reshape((d[0],2,2,d[2])))

    mpo = []
    for i in range(len(mps)):
        A = ncon([mps[i],mps[i]], ((-1,-3,1,-5), (-2,-4,1,-6)))
        d = A.shape
        A = A.reshape((d[0]*d[1], d[2], d[3], d[4]*d[5]))
        mpo.append (A)
    return mpo



if __name__ == '__main__':
    N = 8

    x1,x2 = -2,2

    Ndx = 2**N-1
    rescale = (x2-x1)/Ndx
    shift = x1

    H1, L1, R1 = get_H_SHO (N, rescale)

    H, L, R = H_two_particle2 (H1, L1, R1)

    U = get_swap_U()
    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)
    # Rotate H to the parity basis
    H = ut.applyLocalRot_mpo (H, U)

    # Get initial state MPS
    psi = []
    for i in range(N):
        if i == 0:
            A = np.array([0.,0.,0.,1.])
        else:
            A = (1./3**0.5) * np.array([1.,1.,1.,0.])
        A = A.reshape((1,4,1))
        psi.append(A)

    #
    H, L, R = qn.set_mpo_quantum_number (H, L, R)

    # Define the physical bond
    ii = cytnx.Bond(cytnx.BD_IN, [[0],[1]], [3,1], [cytnx.Symmetry.Zn(2)])
    # Create MPS
    psi = twut.make_product_mps (N)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*20 + [4]*20 + [8]*20 + [16]*20 + [32]*20# + [64]*40# + [128]*40
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)


    # Run dmrg
    psi, ens = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)
    plt.plot(range(len(ens)),ens)
    plt.yscale('log')
    plt.show()

    # ------- Compute the single-particle correlation function -----------
    psi = ut.mps_to_nparray (psi)
    psi = ut.applyLocal_mps (psi, U)

    corr = corr_matrix (psi)
    Lcorr = np.array([1.])
    Rcorr = np.array([1.])
    # Target the largest occupations
    corr[0] *= -1.
    corr, Lcorr, Rcorr = ut.compress_mpo (corr, Lcorr, Rcorr, cutoff=1e-12)

    maxdims = [2]*10 + [4]*10 + [8]*20 + [16]*40

    corr, Lcorr, Rcorr = ut.mpo_to_uniten (corr, Lcorr, Rcorr)
    phi = ut.generate_random_MPS_nparray (N, d=2, D=2)
    phi = ut.mps_to_uniten (phi)

    phi, occ = dmrg.dmrg (phi, corr, Lcorr, Rcorr, maxdims, cutoff)

    phi1 = ut.generate_random_MPS_nparray (N, d=2, D=2)
    phi1 = ut.mps_to_uniten (phi1)
    phi1, occ1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi], weights=[20])

    npphi1 = mpsut.to_npMPS (phi)
    npphi2 = mpsut.to_npMPS (phi1)
    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs, rescale, shift)

    # First particle
    ys1 = [ptut.get_ele_mps (npphi1, bx) for bx in bxs]

    # Second particle
    ys2 = [ptut.get_ele_mps (npphi2, bx) for bx in bxs]

    e1 = npmps.inner_MPO (npphi1, npphi1, H1, L1, R1)
    e2 = npmps.inner_MPO (npphi2, npphi2, H1, L1, R1)
    print(e1,e2,e1+e2)

    fig, ax = plt.subplots()
    ax.plot (xs, ys1, marker='.', ls='None')
    ax.plot (xs, ys2, marker='.', ls='None')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\phi(x)$')
    ps.set(ax)
    fig.savefig('phi.pdf')
    plt.show()

