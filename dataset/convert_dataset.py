"""
yolo用のデータセットをyolosで使えるように変換します

python convert_dataset.py -o [output json file] -l [label dir]

./labelと./rawを元に、train用のjsonファイルを作成します
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path

def format_json(img_path, boxes):
    return {
        "image_path": img_path,
        "labels": {
            "class_labels": [0] * len(boxes),
            "boxes": boxes,
            }
        }

def get_boxes(f):
    b = []
    with open(f, "r") as f:
        for l in f.readlines():
            b.append([float(v) for v in l.strip().split(" ")[1:]])
    return b


parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--image', help="image dir", required=True)
parser.add_argument('-l', '--label', help="label dir", required=True)
parser.add_argument('-o', '--output', help="output file name (ex. d.json)", required=True)
args = parser.parse_args() 

    
# save_file=sys.argv[-1]
# label_dir = "./label/*"
# img_dir = "./raw/*"

save_file=args.output
label_dir = args.label
# img_dir = args.image

label_files = Path(label_dir).glob("*")
# img_files = Path(img_dir).glob("*")

with open(save_file, 'w') as f:
    # random.shuffle(list(label_files))
    for label in label_files:
        name = os.path.splitext(os.path.basename(label))[0]
        if name == "classes":
            continue
        if name == ".DS_Store":
            continue
        
        img_path = "./dataset/raw/{}.jpg".format(name)
        boxes = get_boxes(label)
        j = format_json(img_path, boxes)
        json.dump(j, f)
        f.write('\n')
