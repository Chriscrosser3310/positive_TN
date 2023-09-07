import os
import sys
from time import sleep

script_directory = os.path.dirname(os.path.abspath(sys.argv[0])) + sys.argv[1]

with open(script_directory, 'r') as f:
    last_line = f.read()
while 'done' not in last_line:
    os.system(f"mail -s 'positive_TN' jchen9@caltech.edu < {script_directory}")
    print("email sent")
    sleep(3600)