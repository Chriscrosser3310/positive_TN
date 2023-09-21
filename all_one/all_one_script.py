import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from positive_TN_src import *

d = int(sys.argv[1])
nlist = eval(sys.argv[2])
plist = np.linspace(0, 2/d**3, int(sys.argv[3]), endpoint=True)
repeat = int(sys.argv[4])
width_mode = sys.argv[5]
mode = sys.argv[6]

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
npz_name = f'{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_{sys.argv[5]}_{sys.argv[6]}'
npz_directory = script_directory + f'/{npz_name}.npz'
with open(npz_directory, 'a+') as f:
    pass
print(npz_directory)

avg_table, std_table = avg_entropy_nplist(nlist, plist, bdim=d, repeat=repeat, mode=mode, prt=True, filename=npz_name, width_mode=width_mode)
with open(npz_directory, 'wb') as f:
    np.savez(f, d=[d], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table)
print("done")

os.system(f"mail -a {npz_name}.npz -s 'positive_TN: {npz_name}' jchen9@caltech.edu < {npz_name}.txt")