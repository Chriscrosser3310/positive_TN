from positive_TN_src import *
import os
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
npz_directory = script_directory + '/all_one.npz'
with open(npz_directory, 'a+') as f:
    pass

print(npz_directory)

d = int(sys.argv[1])
nlist = eval(sys.argv[2])
plist = np.linspace(0, 2/d**3, int(sys.argv[3]), endpoint=True)
repeat = int(sys.argv[4])

avg_table, std_table = avg_entropy_nplist(nlist, plist, bdim=d, repeat=repeat, mode="all_one", prt=True)

with open(npz_directory, 'wb') as f:
    np.savez(f, d=[d], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table)

print("done")
