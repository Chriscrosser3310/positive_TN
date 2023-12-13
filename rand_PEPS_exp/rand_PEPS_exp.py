import numpy as np
import quimb.tensor as qtn
import scipy.stats as stats
import matplotlib.pyplot as plt

def Haar_rand(size):
    d = np.prod(size)
    U = stats.unitary_group.rvs(d)
    psi = U[0, :]
    return psi.reshape(size)

def rand_instance(Lx, D, phys_dim, chi):

    Ly = 4 * Lx
    #chi = D**3
    peps = qtn.PEPS.from_fill_fn(
        lambda shape: Haar_rand(size=shape),
        Lx + 1, Ly, D, phys_dim=phys_dim
    )

    # make the 2-norm sandwich, this also tags the layers 'KET' and 'BRA'
    norm = peps.make_norm()

    # perform the boundary contraction, using two layer contraction method
    norm.contract_boundary_from_xmin(
        # sweep from 0 to penultimate row
        (0, Lx - 1),
        max_bond=chi,
        layer_tags=('KET', 'BRA'),
        # strip off an overall scalar factor at each step
        equalize_norms=1.0,
        inplace=True,
    )

    # any x_tag < Lx - 1 will select boundary 
    boundary = norm.select('X0')

    # specify the two tensors across the bisection
    ymida = peps.y_tag(peps.Ly // 2 - 1)
    ymidb = peps.y_tag(peps.Ly // 2)

    # get the bond that connects them
    bond, = qtn.bonds(boundary[ymida], boundary[ymidb])

    # canonicalize around one of the tensors
    boundary.canonize_around_(ymida)

    # compute the singular values, of tensor now in canonical form
    s = boundary[ymida].singular_values([bond])

    s_normalized = s / boundary.norm()
    renyi_2 = -np.log(np.sum(s_normalized**4))
    return s_normalized, renyi_2

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
    
    D = int(sys.argv[1])
    phys_dim = int(sys.argv[2])
    Wlist = eval(sys.argv[3])
    repeat = int(sys.argv[4])
    chi = int(sys.argv[5])

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    npz_name = f'PEPS_D={sys.argv[1]}_physdim={sys.argv[2]}_W={sys.argv[3]}_repeat={sys.argv[4]}_chi={sys.argv[5]}'
    npz_directory = script_directory + f'/{npz_name}.npz'
    with open(npz_directory, 'a+') as f:
        pass
    print(npz_directory)

    svals = []
    es = []
    avgs = []
    stds = []

    for (i, W) in enumerate(Wlist):
        svals.append([])
        es.append([])
        for _ in range(repeat):
            s, e = rand_instance(W, D, phys_dim, chi)
            svals[i].append(s)
            es[i].append(e)
            #with open(npz_directory, 'wb') as f:
                #np.savez(f, D=D, phys_dim=phys_dim, Wlist=Wlist, repeat=repeat, chi=chi, svals=svals, es=es, avgs=avgs, stds=stds)
        avgs.append(np.average(es[i]))
        stds.append(np.std(es[i]))
        #with open(npz_directory, 'wb') as f:
            #np.savez(f, D=D, phys_dim=phys_dim, Wlist=Wlist, repeat=repeat, chi=chi, svals=svals, es=es, avgs=avgs, stds=stds)
        print(f"======== W={W} done ========")
    
    with open(npz_directory, 'wb') as f:
        np.savez(f, D=D, phys_dim=phys_dim, Wlist=Wlist, repeat=repeat, chi=chi, svals=svals, es=es, avgs=avgs, stds=stds)
    print("done")

    os.system(f"mail -a {npz_name}.npz -s 'rand_PEPES_exp: {npz_name}' jchen9@caltech.edu")