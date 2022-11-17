"""
python rename.py [start num] [target dir]

target dirのファイルをstart_numから連番の数字にrenameします

"""


import glob
import sys
import os
from pathlib import Path


if len(sys.argv) != 3:
    print("must set start num")
    print("must set target dir")
    print("python make_set.py [start num] [target dir]")
    exit(0)

start_num = int(sys.argv[-2])
target_dir = sys.argv[-1]

files = Path(target_dir).glob("*")
for i,f in enumerate(files):
    t = Path(f.parent)/Path("{}.jpg".format(str(i+start_num+1).ljust(6, '0')))
    print("{} to {}".format(f, t))
    os.rename(f, t)
print("-----")

files = Path(target_dir).glob("*")
for i,f in enumerate(files):
    t = Path(f.parent)/Path("{}.jpg".format(str(i+start_num+1).rjust(6, '0')))
    print("{} to {}".format(f, t))
    os.rename(f, t) 
    

