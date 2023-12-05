import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from positive_TN_src import *

#python -u all_one_script.py d nlist ptuple repeat width_mode site_mode entropy_type cutoff

d = int(sys.argv[1])
nlist = eval(sys.argv[2])
ptuple = eval(sys.argv[3])
plist = np.linspace(ptuple[0], ptuple[1]/d**3, int(ptuple[2]), endpoint=True) #np.linspace(0, 2/d**3, int(sys.argv[3]), endpoint=True)
repeat = int(sys.argv[4])
width_mode = sys.argv[5]
site_mode = eval(sys.argv[6])
entropy_type = sys.argv[7]
cutoff = float(sys.argv[8])

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
npz_name = f'{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_{sys.argv[5]}_{site_mode[0]},{site_mode[1]}_{sys.argv[7]}_{sys.argv[8]}'
npz_directory = script_directory + f'/{npz_name}.npz'
with open(npz_directory, 'a+') as f:
    pass
print(npz_directory)

avg_table, std_table, raw_data = avg_entropy_nplist(nlist, 
                                          plist, 
                                          bdim=d, 
                                          repeat=repeat, 
                                          entropy_type=entropy_type, 
                                          site_mode=site_mode, 
                                          prt=True, 
                                          filename=npz_name, 
                                          width_mode=width_mode, 
                                          cutoff=cutoff)
with open(npz_directory, 'wb') as f:
    np.savez(f, d=[d], nlist=nlist, plist=plist, avg_table=avg_table, std_table=std_table, raw_data=raw_data)
print("done")

os.system(f"mail -a {npz_name}.npz -s 'positive_TN: {npz_name}' jchen9@caltech.edu < {npz_name}.txt")