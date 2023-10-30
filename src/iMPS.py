from autoray import numpy
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from matplotlib import pyplot as plt
import scipy.stats as stats
import scipy
import itertools
from stats_model import gamma, omega

#https://journals.aps.org/prb/pdf/10.1103/PhysRevB.78.155117
# G == Gamma, l == lambda
def canonicalize_iMPS(G, l, cutoff=1E-15, maxbond=None, normalize=False):

    assert G.shape[0] == G.shape[1] == l.shape[0] == l.shape[1], "Gamma, lambda dimension mismatch"
    bdim, pdim = G.shape[0], G.shape[2]

    #R = np.einsum("abi,bc,dei,ef->adcf", G, l, np.conj(G), np.conj(l))
    R_temp = np.swapaxes(np.tensordot(G, l, (1, 0)), 1, 2)
    R = np.swapaxes(np.tensordot(R_temp, R_temp.conj(), (2, 2)), 1, 2)
    R = np.reshape(R, (bdim**2, bdim**2))

    #L = np.einsum("ab,bci,de,efi->adcf", l, G, np.conj(l), np.conj(G))
    L_temp = np.tensordot(l, G, (1, 0))
    L = np.swapaxes(np.tensordot(L_temp, L_temp.conj(), (2, 2)), 1, 2)
    L = np.reshape(L, (bdim**2, bdim**2))

    #print(R.shape, L.shape)

    #VR_dim = bdim**2
    R_evals, R_evecs = np.linalg.eig(R)
    VR = R_evecs[:, R_evals.argmax()]
    VR = np.reshape(VR, (bdim, bdim))

    #VL_dim = bdim**2
    LT_evals, LT_evecs = np.linalg.eig(L.T)
    VL = LT_evecs[:, LT_evals.argmax()]
    VL = np.reshape(VL, (bdim, bdim))

    D, W = np.linalg.eigh(VR)
    if any(D < 0):
        D = np.abs(D)
    D = np.diag(D)
    X = W @ np.sqrt(D)

    D, W = np.linalg.eigh(VL)
    if any(D < 0):
        D = np.abs(D)
    D = np.diag(D)
    Y = W @ np.sqrt(D)

    YT_l_X = Y.T @ l @ X
    U, lp, V = np.linalg.svd(YT_l_X)
    if type(maxbond) == int:
        lp = lp[0:maxbond]
    else:
        lp = lp[lp >= cutoff]
    U = U[:, 0:len(lp)]
    V = V[0:len(lp), :]
    lp = np.diag(lp)

    Gp = np.zeros((lp.shape[0], lp.shape[1], pdim), dtype="complex128")
    for i in range(pdim):
        Gp[:, :, i] = V @ np.linalg.pinv(X) @ G[:, :, i] @ np.linalg.pinv(Y.T) @ U

    if normalize:
        lp = lp/np.sqrt(R_evals.max())
    
    return Gp, lp

def steady_iMPS(mps_G, mps_l, mpo_G, mpo_l=None, cutoff=1E-15, maxbond=None, maxiter=10):

    mpo_bdim = mpo_G.shape[0]
    if mpo_l is None:
        mpo_l = np.eye(mpo_bdim, dtype="complex128")
    pdim =  mps_G.shape[2]

    assert mps_G.shape[0] == mps_G.shape[1] == mps_l.shape[0] == mps_l.shape[1], "MPS bond dimension mismatch"
    assert mps_G.shape[0] == mpo_G.shape[1] == mpo_l.shape[0] == mpo_l.shape[1], "MPO bond dimension mismatch"
    assert mps_G.shape[2] == mpo_G.shape[2], "MPS, MPO physical dimension mismatch"

    mps_bdim = mps_G.shape[0]
    G_old, l_old = mps_G, mps_l
    G_new = np.swapaxes(np.tensordot(G_old, mpo_G, (2, 3)), 1, 2).reshape((mps_bdim*mpo_bdim, mps_bdim*mpo_bdim, pdim))
    #G_new = np.einsum("lrd,jkud->ljrku", G_old, mpo_G).reshape((mps_bdim*mpo_bdim, mps_bdim*mpo_bdim, pdim))
    l_new = np.kron(l_old, mpo_l)
    G_new, l_new = canonicalize_iMPS(G_new, l_new, cutoff=cutoff, maxbond=maxbond, normalize=True)
  
    def GL_same(G1, l1, G2, l2):
        if G1.shape != G2.shape:
            return False
        if l1.shape != l2.shape:
            return False
        if not np.all(np.isclose(G1, G2)):
            return False
        if not np.all(np.isclose(l1, l2)):
            return False
        return True
    
    iter = 0
    while (not GL_same(G_new, l_new, G_old, l_old)) and iter < maxiter:
        print(f"iter {iter}")
        iter += 1

        G_old, l_old = G_new, l_new
        mps_bdim = G_old.shape[0]
        G_new = np.swapaxes(np.tensordot(G_old, mpo_G, (2, 3)), 1, 2).reshape((mps_bdim*mpo_bdim, mps_bdim*mpo_bdim, pdim))
        #G_new = np.einsum("lrd,jkud->ljrku", G_old, mpo_G).reshape((mps_bdim*mpo_bdim, mps_bdim*mpo_bdim, pdim))
        l_new = np.kron(l_old, mpo_l)
        G_new, l_new = canonicalize_iMPS(G_new, l_new, cutoff=cutoff, maxbond=maxbond, normalize=True)
    
    return G_new, l_new

def pTN_steady(d, mu, cutoff=1E-15, maxbond=7, maxiter=10, return_boundary_num=True):
    g = gamma(d)
    sg = scipy.linalg.sqrtm(g)
    
    mps_l = sg
    mps_G = np.einsum("lrp,pk->lrk", omega(d, mu, 3), sg)
    mpo_G = np.einsum("ijkl,ia,jb,kc,ld->abcd", omega(d, mu, 4), sg, sg, sg, sg)

    Gp, lp = steady_iMPS(mps_G, mps_l, mpo_G, cutoff=cutoff, maxbond=maxbond, maxiter=maxiter)
    
    if return_boundary_num:
        R0 = Gp[:, :, 0] @ lp
        R1 = Gp[:, :, 1] @ lp
        L1 = lp @ Gp[:, :, 1]
        L0 = lp @ Gp[:, :, 0]

        R0_evals, R0_evecs = np.linalg.eig(R0)
        etaR0 = R0_evals.max()
        vecR0 = R0_evecs[:, R0_evals.argmax()]

        R1_evals, R1_evecs = np.linalg.eig(R1)
        etaR1 = R1_evals.max()
        vecR1 = R1_evecs[:, R1_evals.argmax()]

        L1T_evals, L1T_evecs = np.linalg.eig(L1.T)
        etaL1 = L1T_evals.max()
        vecL1 = L1T_evecs[:, L1T_evals.argmax()]

        L0T_evals, L0T_evecs = np.linalg.eig(L0.T)
        etaL0 = L0T_evals.max()
        vecL0 = L0T_evecs[:, L0T_evals.argmax()]

        return np.abs((vecL1.T @ lp @ vecR0)/(vecL0.T @ lp @ vecR0)), \
               np.abs((vecL1.T @ lp @ vecR0)/(vecL1.T @ lp @ vecR1)), \
               np.abs((vecL0.T @ lp @ vecR1)/(vecL0.T @ lp @ vecR0)), \
               np.abs((vecL0.T @ lp @ vecR1)/(vecL1.T @ lp @ vecR1)), \
               (etaR0 * etaL1)/(etaR0 * etaL0), \
               (etaR0 * etaL1)/(etaR1 * etaL1), \
               (etaR1 * etaL0)/(etaR0 * etaL0), \
               (etaR1 * etaL0)/(etaR1 * etaL1)

    else:
        return Gp, lp

if __name__ == "__main__":
    """
    o = omega(4, 1, 3)
    g = gamma(4)

    l = scipy.linalg.sqrtm(g)
    G = np.einsum("lrp,pk->lrk", o, l)

    G_old, l_old = G, l
    G_new, l_new = canonicalize_iMPS(G, l, 0, None, True)

    R = np.einsum("abi,bc,dei,ef->adcf", G_new, l_new, np.conj(G_new), np.conj(l_new))
    print(np.around(np.einsum("abcc -> ab", R), 4))

    L = np.einsum("ab,bci,de,efi->adcf", l_new, G_new, np.conj(l_new), np.conj(G_new))
    print(np.around(np.einsum("aabc -> bc ", L), 4))
    """

    print(np.min(pTN_steady(7, 1, maxbond=4, maxiter=20)[0:4]))