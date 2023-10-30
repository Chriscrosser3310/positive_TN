import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
from iMPS import *

#python -u iMPS_script.py dtuple mutuple

dtuple = eval(sys.argv[1])
mutuple = eval(sys.argv[2])
cutoff = float(sys.argv[3])
maxbond = int(sys.argv[4])
maxiter = int(sys.argv[5])

dlist = np.linspace(dtuple[0], dtuple[1], dtuple[2], endpoint=True)
mulist = np.linspace(mutuple[0], mutuple[1], mutuple[2], endpoint=True)

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
npz_name = f'iMPS_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_{sys.argv[5]}'
npz_directory = script_directory + f'/{npz_name}.npz'
with open(npz_directory, 'a+') as f:
    pass
print(npz_directory)

vals = np.zeros((len(dlist), len(mulist)))
for (i, d) in enumerate(dlist):
    for (j, mu) in enumerate(mulist):
        val = np.min(pTN_steady(d, mu, cutoff=cutoff, maxbond=maxbond, maxiter=maxiter)[0:4])
        vals[i, j] = val
        np.savez(npz_directory, dlist=dlist, mulist=mulist, vals=vals)
print("done")

os.system(f"mail -a {npz_name}.npz -s 'positive_TN: {npz_name}' jchen9@caltech.edu")