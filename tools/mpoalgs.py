import numpy_dmrg as dmrg

# Compute mpo|mps> approximately
# It works better if the result is close to mps0
def fit_apply_MPO (mpo, mps, mps0, numCenter, nsweep=1, maxdim=100000000, cutoff=0.):
    # Check the MPS and the MPO
    assert (len(mpo) == len(mps) == len(mps0))
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    npmps.check_MPS_links (mps0)

    psi = copy.copy(psi)

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    N = len(psi)
    ranges = get_sweeping_sites (N, numCenter)

    # Get the environment tensors
    LR = LR_envir_tensors_mpo (N)
    LR.update_LR (psi, psi, H, 0)

    ens, terrs = [], []
    for k in range(nsweep):                                                            # For each sweep
        for lr in [0,1]:
            for p in ranges[lr]:
                #
                #         2                   2      3
                #         |                   |______|
                #    1 ---O--- 3   or   1 ---(________)--- 4
                phi = get_eff_psi (psi, p, numCenter)
                dims = phi.shape
                phi = phi.reshape(-1)

                # Update the environment tensors
                if numCenter == 2:
                    LR.update_LR (psi, psi, H, p, p+1)
                elif numCenter == 1:
                    LR.update_LR (psi, psi, H, p)

                # Define the effective Hamiltonian
                effH = get_eff_H (LR, H, p, numCenter)

                # Find the new state for the current bond
                phi =  (effH, phi, k=krylovDim)
                phi = phi.reshape(dims)
                phi = phi / np.linalg.norm(phi)

                # Update tensors
                toRight = (lr==0)
                if numCenter == 2:
                    psi[p], psi[p+1], err = npmps.truncate_svd2 (phi, rowrank=2, toRight=toRight, cutoff=cutoff)
                    terr += err
                elif numCenter == 1:
                    psi = update_mps_1site (phi, p, psi, toRight)
                else:
                    raise Exception

        if verbose:
            print('Sweep',k,', chi='+str(chi),', maxdim='+str(max(npmps.MPS_dims(psi))))
            print('\t','energy =',en, terr)

        ens.append(en);
        terrs.append (terr/N_update)
    return psi, ens, terrs

def exact_apply_MPO (mpo, mps):
    assert len(mpo) == len(mps)
    check_MPO_links(mpo)
    check_MPS_links(mps)

    mpo = copy.copy(mpo)

    A1 = ncon([mps[0], mpo[0]], ((-1,1,-4),(-2,-3,1,-5)))
    dl1,dl2,di,dr1,dr2 = A1.shape
    A1 = A1.reshape((1,di,dr1,dr2))
    res = []
    for i in range(1,len(mps)):
        A2 = ncon([mps[i], mpo[i]], ((-1,1,-4),(-2,-3,1,-5)))
        AA = ncon([A1,A2], ((-1,-2,1,2),(1,2,-3,-4,-5)))
        dl,di1,di2,dr1,dr2 = AA.shape
        AA = AA.reshape((dl*di1, di2*dr1*dr2))

        U, S, Vh = np.linalg.svd (AA, full_matrices=False)
        A = (U*S).reshape((dl,di1,-1))
        A1 = Vh.reshape((-1,di2,dr1,dr2))

        res.append(A)
    dl1,di,dr1,dr2 = A1.shape
    A = A1.reshape((dl1,di,1))
    res.append(A)
    return res

