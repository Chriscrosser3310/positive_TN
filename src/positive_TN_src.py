from autoray import numpy
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import matplotlib
from matplotlib import pyplot as plt
import scipy.stats as stats
import os
import sys
import math

"""
def random_MPS(n, bdim=2, r=(-0.5, 0.5)):
    rmin, rmax = r
    scalar = (rmax - rmin)
    shift = rmin

    def rand_gen(shape):
        np.random.random(shape) * scalar + shift

    arrays = [None for _ in range(n)]
    arrays[0] = rand_gen((bdim, bdim))
    arrays[-1] = rand_gen((bdim, bdim))
    for i in range(1, n-1):
        arrays[i] = rand_gen((bdim, bdim, bdim))
    
    return qtn.MatrixProductState(arrays)

def random_MPO(n, bdim=2, r=(-0.5, 0.5)):
    rmin, rmax = r
    scalar = (rmax - rmin)
    shift = rmin

    def rand_gen(shape):
        np.random.random(shape) * scalar + shift

    arrays = [None for _ in range(n)]
    arrays[0] = rand_gen((bdim, bdim, bdim))
    arrays[-1] = rand_gen((bdim, bdim, bdim))
    for i in range(1, n-1):
        arrays[i] = rand_gen((bdim, bdim, bdim, bdim))
    
    return qtn.MatrixProductOperator(arrays)

def random_PEPS_config(n, bdim=2, r=(-0.5, 0.5)):
    PEPS_config = random_MPS(n, bdim, r)
    for _ in range(n-2):
        PEPS_config = PEPS_config & random_MPO(n, bdim, r)
    PEPS_config = PEPS_config & random_MPS(n, bdim, r).H
    return PEPS_config

random_PEPS_config(4)
"""

def TN2D_rand(n, bdim=2, r=(-0.5, 0.5)):
    rmin, rmax = r
    return qtn.TN2D_rand(n, n, bdim, dist="uniform", loc=rmin, scale=rmax-rmin)

def move_loc_test(n, locs, scales, bdim=2, dist="normal", repeat=10):
    max_bonds = []
    central_bond_lim = 2**(n-1)
    for (loc, scale) in zip(locs, scales):
        print(loc)
        loc_max_bonds = []
        for _ in range(repeat):
            print("-", end="")
            tn = qtn.TN2D_rand(n, n, bdim, dist=dist, loc=loc, scale=scale)
            tn_val = tn.contract_boundary_from_xmin((0, n), max_bond=central_bond_lim, cutoff=0).contract()
            tn_val_app = 0
            max_bond = 1
            while (np.abs(tn_val_app - tn_val)/np.abs(tn_val) >= 1E-15) and (max_bond < central_bond_lim):
                max_bond += 1
                tn_val_app = tn.contract_boundary_from_xmin((0, n), max_bond=max_bond, cutoff=0).contract()
            #print(tn_val_app, tn_val)
            loc_max_bonds.append(max_bond)
        max_bonds.append(np.mean(loc_max_bonds))
        print()
    return max_bonds

def rand_MPO(n, rand_fn, bdim=2, tensorwise=True):
    arrays = [None for _ in range(n)]
    if tensorwise:
        for r in range(1, n-1):
            arrays[r] = rand_fn()
        arrays[0] = rand_fn()[0, :, :, :]
        arrays[-1] = rand_fn()[:, :, :, 0]
    else:
        for r in range(1, n-1):
            t = np.zeros((bdim, bdim, bdim, bdim))
            for i in range(bdim):
                for j in range(bdim):
                    for k in range(bdim):
                        for l in range(bdim):
                            t[i, j, k, l] = rand_fn()
            arrays[r] = t
        for r in range(-1, 1):
            t = np.zeros((bdim, bdim, bdim))
            for i in range(bdim):
                for j in range(bdim):
                    for k in range(bdim):
                        t[i, j, k] = rand_fn()
            arrays[r] = t
    return qtn.MatrixProductOperator(arrays)

def rand_MPS(n, rand_fn, bdim=2, tensorwise=True):
    arrays = [None for _ in range(n)]
    if tensorwise:
        for r in range(1, n-1):
            arrays[r] = rand_fn()
        arrays[0] = rand_fn()[0, :, :]
        arrays[-1] = rand_fn()[:, :, 0]
    else:
        for r in range(1, n-1):
            t = np.zeros((bdim, bdim, bdim))
            for i in range(bdim):
                for j in range(bdim):
                    for k in range(bdim):
                        t[i, j, k] = rand_fn()
            arrays[r] = t
        for r in range(-1, 1):
            t = np.zeros((bdim, bdim))
            for i in range(bdim):
                for j in range(bdim):
                    t[i, j] = rand_fn()
            arrays[r] = t
    return qtn.MatrixProductState(arrays)

def boundary_mps(n, p, bdim=2, site_mode="all_one", width_mode="full"):
    """
    #rand_fn = lambda: (1-p)*(np.random.random()*2-1) + p   
    #rand_fn = lambda: (1-p)*(np.random.normal()) + p
    #rand_fn = lambda: (1-p)*((np.random.random()*2-1) + 1j*(np.random.random()*2-1)) + p

    mps = rand_MPS(n, rand_fn_mps, bdim)
    mpos = []
    for _ in range(int(np.ceil(n/2)) - 1):
    #for _ in range(n-1):
        mpos.append(rand_MPO(n, rand_fn_mpo, bdim))
    """
    rand_fn_mps = None
    rand_fn_mpo = None
    
    if site_mode == "all-one":
        p_mat = np.ones((bdim, bdim, bdim, bdim))

        rand_fn_mps = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim))[0, :, :, :] + p*p_mat[0, :, :, :]
        rand_fn_mpo = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim)) + p*p_mat

    if site_mode == "all-one-ortho":
        p_mat = np.ones((bdim, bdim, bdim, bdim))

        rand_fn_mps = lambda: stats.ortho_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim))[0, :, :, :] + p*p_mat[0, :, :, :]
        rand_fn_mpo = lambda: stats.ortho_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim)) + p*p_mat

    elif site_mode == "rand-rank-one":
        u1 = stats.unitary_group.rvs(bdim)
        u2 = stats.unitary_group.rvs(bdim)
        u3 = stats.unitary_group.rvs(bdim)
        u4 = stats.unitary_group.rvs(bdim)

        p_mat = np.ones((bdim, bdim, bdim, bdim))
        p_mat = np.einsum("ijkl, ih -> hjkl", p_mat, u1)
        p_mat = np.einsum("ijkl, jh -> ihkl", p_mat, u2)
        p_mat = np.einsum("ijkl, kh -> ijhl", p_mat, u3)
        p_mat = np.einsum("ijkl, lh -> ijkh", p_mat, u4)
    
        rand_fn_mps = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim))[0, :, :, :] + p*p_mat[0, :, :, :]
        rand_fn_mpo = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim)) + p*p_mat
    
    elif site_mode == "rand-positive-sphere":

        u = stats.unitary_group.rvs(bdim**4)
        p_mat = np.ones(bdim**4)
        p_mat = u @ p_mat
        p_mat = p_mat.reshape((bdim, bdim, bdim, bdim))
        p_mat = np.abs(p_mat)

        rand_fn_mps = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim))[0, :, :, :] + p*p_mat[0, :, :, :]
        rand_fn_mpo = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim)) + p*p_mat

    elif site_mode == "rand-positive-renorm":

        p_mat = np.random.random((bdim, bdim, bdim, bdim))
        p_mat = p_mat*np.sqrt(bdim**4/np.sum(p_mat ** 2))

        rand_fn_mps = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim))[0, :, :, :] + p*p_mat[0, :, :, :]
        rand_fn_mpo = lambda: stats.unitary_group.rvs(bdim**4)[0, :].reshape((bdim, bdim, bdim, bdim)) + p*p_mat
    
    # site_mode == ("rand-PSD", r) where r is an integer indicating the physical-bond rank
    # p = [l, u]
    elif site_mode[0] == "rand-PSD":
        r = site_mode[1]
        def temp_mpo():
            U = stats.unitary_group.rvs(bdim**4)
            Ud = U.conj().T
            
            #D = np.diag(np.random.uniform(0, p, min(bdim**4, r)))
            #M = U[:, 0:r] @ D @ Ud[0:r, :]
            
            M = U[:, 0:r] @ Ud[0:r, :]
            T = np.reshape(M, [bdim]*8)
            T = np.einsum("abcdijkl -> aibjckdl", T)
            T = np.reshape(T, [bdim**2]*4)
            return T
        def temp_mps():
            return temp_mpo()[0, :, :, :]
        
        rand_fn_mps = temp_mps
        rand_fn_mpo = temp_mpo
    
    # site_mode == ("rand-PSD-Haar", r) where r is an integer indicating the physical-bond rank
    # p = [l, u]
    elif site_mode[0] == "rand-PSD-Haar":
        r = site_mode[1]
        def temp_mpo():
            U = stats.unitary_group.rvs(bdim**4*r)
            psi = U[0, :]
            psi_reshape = np.reshape(psi, (bdim**4, r))
            M =  psi_reshape @ psi_reshape.conj().T
            T = np.reshape(M, [bdim]*8)
            T = np.einsum("abcdijkl -> aibjckdl", T)
            T = np.reshape(T, [bdim**2]*4)
            return T
        def temp_mps():
            return temp_mpo()[0, :, :, :]
        
        rand_fn_mps = temp_mps
        rand_fn_mpo = temp_mpo

    # site_mode == ("rand-PSD-gaussian", r) where r is an integer indicating the physical-bond rank
    # p = [l, u]
    elif site_mode[0] == "rand-PSD-gaussian":
        r = site_mode[1]
        def temp_mpo():
            A = np.random.normal(loc=0, scale=1, size=(bdim**4, 2)) + 1j*np.random.normal(loc=0, scale=1, size=(bdim**4, 2))
            T = A @ A.T.conj()
            T = np.reshape(T, [bdim]*8)
            T = np.einsum("abcdijkl -> aibjckdl", T)
            T = np.reshape(T, [bdim**2]*4)
            return T
        def temp_mps():
            return temp_mpo()[0, :, :, :]
        
        rand_fn_mps = temp_mps
        rand_fn_mpo = temp_mpo

    elif site_mode[0] == "rand-PSD-diag":
        r = site_mode[1]
        def temp_mpo():
            U = stats.unitary_group.rvs(bdim**4)
            Ud = U.conj().T
            D = np.diag(np.random.uniform(0, p, min(bdim**4, r)))
            M = U[:, 0:r] @ D @ Ud[0:r, :]
            T = np.reshape(M, [bdim**2]*4)
            return T
        def temp_mps():
            return temp_mpo()[0, :, :, :]
        
        rand_fn_mps = temp_mps
        rand_fn_mpo = temp_mpo

    elif site_mode[0] == "rand-PSD-positive":
        r = site_mode[1]
        p2 = site_mode[2]
        def temp_mpo():
            U = stats.unitary_group.rvs(bdim**4)
            Ud = U.conj().T
            D = np.diag(np.random.uniform(0, p, min(bdim**4, r)))
            M = U[:, 0:r] @ D @ Ud[0:r, :]
            T = np.reshape(M, [bdim]*8)
            T = np.einsum("abcdijkl -> aibjckdl", T)
            T = np.reshape(T, [bdim**2]*4)
            T += p2 * np.ones([bdim**2]*4)
            return T
        def temp_mps():
            return temp_mpo()[0, :, :, :]
        
        rand_fn_mps = temp_mps
        rand_fn_mpo = temp_mpo

    mps = rand_MPS(n, rand_fn_mps, bdim, tensorwise=True)
    mpos = []
    if width_mode == "full":
        it = range(n-1)
    elif width_mode == "half":
        it = range(int(np.ceil(n/2)) - 1)
    elif width_mode == "double":
        it = range(2*n-1)
    elif width_mode == "quarter":
        it = range(int(np.ceil(n/4)) - 1)
    else:
        it = range(int(width_mode))
    for _ in it:
        mpo = rand_MPO(n, rand_fn_mpo, bdim, tensorwise=True)
        mpos.append(mpo)
    
    return mps, mpos

 

"""
def avg_entropy(n, p, bdim=2, repeat=10, mode="all_one", entropy_type="renyi-2", width_mode="full"):
    es = []
    for _ in range(repeat):
        es.append(boundary_mps_entropy(n, p, bdim, mode, entropy_type, width_mode))

    return np.average(es), np.std(es)


def avg_entropy_nlist(nlist, p, bdim=2, repeat=10, mode="all_one", prt=False, entropy_type="renyi-2"):
    avgs = []
    stds = []
    if prt:
        print("Finished: ", end="")
    for n in nlist:
        avg, std = avg_entropy(n, p, bdim, repeat, mode, entropy_type)
        avgs.append(avg)
        stds.append(std)
        if prt:
            print(f"{n}", end = " ")
    if prt:
        print()
    return avgs, stds
"""

def avg_entropy_nplist(nlist, 
                       plist, 
                       bdim=2, 
                       repeat=20, 
                       site_mode="all_one", 
                       prt=True, 
                       save_prt=True, 
                       entropy_type="renyi-2", 
                       filename=None, 
                       width_mode="full",
                       cutoff=1E-15):
    
    n_num, p_num = len(nlist), len(plist)
    avg_table = np.array([[np.nan for _ in range(p_num)] for _ in range(n_num)])
    std_table = np.array([[np.nan for _ in range(p_num)] for _ in range(n_num)])
    raw_data = np.array([[[np.nan for _ in range(repeat)] for _ in range(p_num)] for _ in range(n_num)])

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    if filename == None:
        file_directory = script_directory + f"/{site_mode}.npz"
        txt_directory = script_directory + f"/{site_mode}.txt"
    else:
        file_directory = script_directory + f"/{filename}.npz"
        txt_directory = script_directory + f"/{filename}.txt"
    np.savez(file_directory, d=[bdim], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table, raw_data=raw_data)
    for (i, p) in enumerate(plist):
        if prt:
            print(f"-------p = {p}-------")
            print("Finished: ", end="")
        if save_prt:
            with open(txt_directory, "a+") as f:
                f.write(f"-------p = {p}-------\n")
                f.write("Finished: ")
        for (j, n) in enumerate(nlist):
            #avg, std = avg_entropy(n, p, bdim, repeat, site_mode, entropy_type, width_mode)
            #pavgs, pstds = avg_entropy_nlist(nlist, p, bdim, repeat, site_mode, prt, entropy_type)
            es = []
            for _ in range(repeat):
                mps, mpos = boundary_mps(n, p, bdim, site_mode, width_mode)

                e = None
                if entropy_type == "sign-problem":
                    def mp_abs(mp):
                        nsite = len(mps.shape)
                        arrays = []
                        for i in range(nsite):
                            arrays.append(np.abs(mp[i].data))
                        if type(mp) is qtn.tensor_1d.MatrixProductState:
                            return qtn.MatrixProductState(arrays)
                        elif type(mp) is qtn.tensor_1d.MatrixProductOperator:
                            return qtn.MatrixProductOperator(arrays)
                    
                    mps_out = mps
                    #mps_out.normalize()
                    for mpo in mpos:
                        mps_out = mpo.apply(mps_out, compress=True, cutoff=cutoff)
                        #mps_out.normalize()
                    
                    #no sign
                    mps_out_ns = mp_abs(mps)
                    #mps_out_ns.normalize()
                    for mpo in mpos:
                        mps_out_ns = mp_abs(mpo).apply(mps_out_ns, compress=True, cutoff=cutoff)
                        #mps_out_ns.normalize()

                    azero = np.zeros(bdim)
                    azero[0] = 1.
                    azero = np.array([[azero]])
                    mps_zero = qtn.MatrixProductState([azero[0]] + [azero]*(n-2) + [azero[0]])
                    e = np.real((mps_zero @ mps_out) / (mps_zero @ mps_out_ns))
                    #print(f"sign-problem img of contracted value: {np.imag(e)}")
                elif entropy_type == "von-Neumann" or entropy_type.startswith("renyi"):
                    mps_out = mps
                    mps_out.normalize()
                    for mpo in mpos:
                        mps_out = mpo.apply(mps_out, compress=True, cutoff=cutoff)
                        mps_out.normalize()
                    #mps_out.show()

                    if entropy_type == "von-Neumann":
                        e = mps_out.entropy(n//2)
                    elif entropy_type.startswith("renyi"):
                        k = int(entropy_type.split("-")[1])
                        S = mps_out.schmidt_values(n//2, cur_orthog=None, method='svd')
                        S = S[S > 0.0]
                        e = 1/(1-k)*np.log(np.sum(S**k))
                elif entropy_type == "renyi-2-convergence":

                    e_prev = np.inf
                    diff = np.inf

                    for maxdim in range(bdim, bdim**2):
                        mps_out = mps
                        mps_out.normalize()
                        for mpo in mpos:
                            mps_out = mpo.apply(mps_out, compress=True, cutoff=cutoff, maxdim=maxdim)
                            mps_out.normalize()
                        #mps_out.show()

                        S = mps_out.schmidt_values(n//2, cur_orthog=None, method='svd')
                        S = S[S > 0.0]
                        e_new = 1/(1-k)*np.log(np.sum(S**2))

                        diff = np.abs(e_new - e_prev)
                        if diff < 1E-10:
                            break

                        e_prev = e_new
                    
                    print("maxdim:", maxdim)
                    print("renyi-2 diff:", diff)
                    
                    e = e_new

                else:
                    raise("?")
                es.append(e)
            avg, std = np.average(es), np.std(es)
            avg_table[j, i] = avg
            std_table[j, i] = std
            raw_data[j, i] = es
            if prt:
                print(f"{n}", end = " ")
            if save_prt:
                with open(txt_directory, "a+") as f:
                    f.write(f"{n} ")
            np.savez(file_directory, d=[bdim], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table, raw_data=raw_data)
        if prt:
            print()
        if save_prt:
            with open(txt_directory, "a+") as f:
                f.write(f"\n")
    return avg_table, std_table, raw_data

def plot_finite_sim(filenames, error_bar=True, log_scale=False):
    npz_directorys = []
    if type(filenames) == str:
        filenames = [filenames]

    d, nlist, plist, avg_table, std_table = None, None, None, None, None
    for filename in filenames:
        script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        npz_directory = script_directory + f'{filename}'
        npz_directorys.append(npz_directory)
        npz = np.load(npz_directory)

        if d is None:
            d = npz['d'][0]
        else:
            assert d == npz['d'][0], "bond dimension must match"
        
        if plist is None:
            plist = npz['plist']
        else:
            assert all(plist == plist), "plist must match"
                
        if nlist is None:
            nlist = npz['nlist']
        else:
            nlist = np.append(nlist, npz['nlist'])
        
        if avg_table is None:
            avg_table = npz['avg_table']
        else:
            avg_table = np.append(avg_table, npz['avg_table'], axis=0)

        if std_table is None:
            std_table = npz['std_table']
        else:
            std_table = np.append(std_table, npz['std_table'], axis=0)

    """
    fig = plt.figure(figsize=(8, 4), alpha=0)
    fig.patch.set_alpha(1)
    #plt.plot(plist * d**3, avg_table_psd[0, :])
    plt.errorbar(plist * d**3, avg_table[2, :], yerr=std_table[2, :])
    #plt.title("Rank-one test")
    #plt.xlabel("Eigenvalue range lower bound (p in [p, 1])")
    #plt.ylabel("von Neumman entropy")
    #plt.legend(np.round(plist, 2), loc="upper left", ncol=2)
    plt.show()
    """

    scaled_plist = plist*d**3

    x = scaled_plist
    fig, ax = plt.subplots(figsize=(8, 4), alpha=0)
    
    if error_bar:
        for i, n in enumerate(nlist):
            y = avg_table[i, :]
            yerr = std_table[i, :]
            if log_scale:
                y = -np.log(y)
            markers, caps, bars = ax.errorbar(x = x,
                        y = y, 
                        yerr = yerr,
                        label = n,
                        capsize = 5, 
                        elinewidth = 2,
                        markeredgewidth = 7, 
                        capthick = 2)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            #ax.axhline(y=np.log10(d**(n//2)), linestyle='--', label='_nolegend_')
    else:
        for i, n in enumerate(nlist):
            y = avg_table[i, :]
            if log_scale:
                y = -np.log(y)
            ax.plot(x,
                    y, 
                    label = n,
                    markeredgewidth = 7)
            #ax.axhline(y=np.log10(d**(n//2)), linestyle='--', label='_nolegend_')
    
    ax.axhline(y=0 , color='r', linestyle='--', label='_nolegend_')
    plt.xlabel("p*d^3")
    #ax.legend()
    """
    if "half" in filename:
        plt.title(f"Entanglement of d={d} n by n/2 grid")
    elif "half" in filename:
        plt.title(f"Entanglement of d={d} n by 2n grid")
    else:
        plt.title(f"Entanglement of d={d} n by n grid")
    plt.ylabel("Renyi-2 entropy")
    plt.legend([f"n={n}" for n in nlist], loc="upper right", ncol=2)
    """

    return fig, ax

def plot_iMPS(filename):
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    npz_directory = script_directory + f'{filename}'
    npz = np.load(npz_directory)

    X, Y = np.meshgrid(npz['mulist'], npz['dlist'])
    Z = npz['vals']
    #print(Z)

    Z[Z > 1] = 0.0

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False, vmin=-0.01, vmax=1.01)

    ax.set_xlabel('mu')
    ax.set_ylabel('d')
    ax.set_zlabel('vals')

    fig.colorbar(surf, shrink=1, aspect=5)

    return fig, ax


def combine_npz(filenames):
    npz_directorys = []
    if type(filenames) == str:
        filenames = [filenames]

    d, nlist, plist, avg_table, std_table, raw_data = None, None, None, None, None, None
    for filename in filenames:
        #script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        #npz_directory = script_directory + f'{filename}'
        npz_directory = filename
        npz_directorys.append(npz_directory)
        npz = np.load(npz_directory)

        if d is None:
            d = npz['d'][0]
        else:
            assert d == npz['d'][0], "bond dimension must match"
        
        if plist is None:
            plist = npz['plist']
        else:
            assert all(plist == plist), "plist must match"
                
        if nlist is None:
            nlist = npz['nlist']
        else:
            nlist = np.append(nlist, npz['nlist'])
        
        if avg_table is None:
            avg_table = npz['avg_table']
        else:
            avg_table = np.append(avg_table, npz['avg_table'], axis=0)

        if std_table is None:
            std_table = npz['std_table']
        else:
            std_table = np.append(std_table, npz['std_table'], axis=0)

        if raw_data is None:
            raw_data = npz['raw_data']
        else:
            raw_data = np.append(raw_data, npz['raw_data'], axis=0)
    '''
    #script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    filename_split = filenames[0].split("_")
    filename_split[1] = f"[{','.join([str(n) for n in nlist])}]"
    npz_name = "_".join(filename_split) #f"{d}_[{','.join([str(n) for n in nlist])}]_{len(plist)}_50"
    #npz_directory = script_directory + f'/{npz_name}.npz'
    npz_directory = f'/{npz_name}.npz'
    '''
    with open("combined.npz", 'wb') as f:
        np.savez(f, d=[d], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table, raw_data=raw_data)

if __name__ == "__main__":

    '''
    from pathlib import Path
    p = Path(__file__).with_name('file.data')
    with p.open('r') as f:
        strs = f.read().split('\n')
        st = [s.split() for s in strs][:-1]
        sta = np.array([(float(s[0]), float(s[1])) for s in st])
    '''

    """
    plot_npz(["/../all_one/2_[8,10,12,14]_10_50.npz"])
    plot_npz(["/../all_one/3_[8,10,12,14]_10_50.npz"])
    fig, ax = plot_npz(["/../all_one/4_[8,10,12]_10_50.npz"])
    ax.plot(sta[:, 0], -np.log(sta[:, 1]))
    ax.axvline(sta[:, 0][np.argmin(sta[:, 1])], color='r', linestyle='dashed', label = 'transition point')
    plt.show()
    """

    """
    plot_npz(["/../all_one/2_[8,10,12,14,16]_10_50_half.npz"])
    plot_npz(["/../all_one/3_[8,10,12,14,16]_10_50_half.npz"])
    plot_npz(["/../all_one/4_[8,10,12]_10_50_half.npz"])
    
    plt.show()
    """

    plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/2_[20]_[0,8,40]_10_quarter_rand-positive_renyi-2_1e-15.npz")
    plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/3_[20]_[0,8,40]_10_quarter_rand-positive_renyi-2_1e-15.npz")
    plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/2_[20]_[0,8,40]_10_quarter_rand-positive-sphere_renyi-2_1e-15.npz")
    plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/3_[20]_[0,8,40]_10_quarter_rand-positive-sphere_renyi-2_1e-15.npz")
    #plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/4_[20]_[0,8,40]_10_quarter_rand-positive_renyi-2_1e-15.npz")
    #plot_finite_sim("/../key_data/site_mode_comparision/rand-positive/5_[20]_[0,8,40]_10_quarter_rand-positive_renyi-2_1e-15.npz")
    plt.show()

    plot_finite_sim(["/../key_data/site_mode_comparision/3_[10]_[0,8,40]_50_full_all-one_renyi-2_1e-15.npz", 
                     "/../key_data/site_mode_comparision/3_[10]_[0,8,40]_50_full_all-one-ortho_renyi-2_1e-15.npz", 
                     "/../key_data/site_mode_comparision/3_[10]_[0,8,40]_50_full_rand-positive_renyi-2_1e-15.npz",
                     "/../key_data/site_mode_comparision/3_[10]_[0,8,40]_50_full_rand-rank-one_renyi-2_1e-15.npz"])
    plt.ylabel("renyi-2 entropy")
    plt.legend(["unitary", "orthogonal", "random positive", "random rank-one"])
    #plt.show()

    fig1, ax1 = plot_finite_sim("/../key_data/ortho/2_[8,12,16,20,24,28]_[0,2,10]_50_quarter_all-one-ortho_renyi-2_1E-30.npz")
    plt.title("d=2, orthogonal")
    plt.legend(["W=2", "W=3", "W=4", "W=5", "W=6", "W=7"])
    fig2, ax2 = plot_finite_sim("/../key_data/ortho/3_[8,12,16,20,24,28]_[0,2,10]_50_quarter_all-one-ortho_renyi-2_1E-30.npz")
    plt.title("d=3, orthogonal")
    plt.ylabel("renyi-2")
    plt.legend(["W=2", "W=3", "W=4", "W=5", "W=6", "W=7"])

    fig3, ax3 = plot_finite_sim("/../key_data/unitary/2_[8,12,16,20,24,28]_[0,2,10]_50_quarter_all-one_renyi-2_1e-15.npz")
    plt.title("d=2, unitary")
    plt.ylabel("renyi-2")
    plt.legend(["W=2", "W=3", "W=4", "W=5", "W=6", "W=7"])
    fig4, ax4 = plot_finite_sim("/../key_data/unitary/3_[8,12,16,20,24,28]_[0,2,10]_50_quarter_all-one_renyi-2_1e-15.npz")
    plt.title("d=3, unitary")
    plt.ylabel("renyi-2")
    plt.legend(["W=2", "W=3", "W=4", "W=5", "W=6", "W=7"])
    fig4, ax4 = plot_finite_sim(["/../key_data/unitary/4_[8,12,16]_[0,2,10]_50_quarter_all-one_renyi-2_1e-15.npz"])
    plt.title("d=4, unitary")
    plt.ylabel("renyi-2")
    plt.legend(["W=2", "W=3", "W=4"])
    
    fig, ax = plot_iMPS("/../key_data/iMPS/iMPS_[1.5,10,18]_[0.1,2,20]_1e-15_7_10.npz")
    plt.show()

    #fig1, ax1 = plot_finite_sim("/../key_data/sign-problem/2_[8,10,12,14]_[0,8,20]_50_full_all-one_sign-problem_0.npz", log_scale=True, error_bar=False)
    #fig2, ax2 = plot_finite_sim("/../key_data/sign-problem/2_[8,10,12,14]_[0,8,20]_50_full_all-one-ortho_sign-problem_0.npz", log_scale=True, error_bar=False)
    #plt.show()

    #combine_npz(["/../key_data/unitary/3_[8,12,16,20]_(0,2,10)_50_quarter_all-one_renyi-2_1e-15.npz", "/../key_data/unitary/3_[24,28]_(0,2,10)_50_quarter_all-one_renyi-2_1e-15.npz"])