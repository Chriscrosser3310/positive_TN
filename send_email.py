import os
import sys
from time import sleep

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
with open(script_directory + '\\sbatch_buffer.txt', 'r', encoding='utf-16') as f:
    last_line = f.read()
while 'done' not in last_line:
    os.system("mail -s 'positive_TN' jchen9@caltech.edu < sbatch_buffer.txt")
    print("email sent")
    sleep(60)