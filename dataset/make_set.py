"""
python make_set.py [output json file]

./labelと./rawを元に、train用のjsonファイルを作成します
"""

import glob
import json
import os
import sys
import random

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


if len(sys.argv) != 2:
    print("python make_set.py [json file]")
    exit(0)
    
save_file=sys.argv[-1]

label_dir = "./label/*"
img_dir = "./raw/*"

label_files = glob.glob(label_dir)
img_files = glob.glob(img_dir)

with open(save_file, 'w') as f:
    random.shuffle(label_files)
    for label in label_files:
        name = os.path.splitext(os.path.basename(label))[0]    
        if name != "classes":
            img_path = "./dataset/raw/{}.jpg".format(name)
            boxes = get_boxes("./label/{}.txt".format(name))
            j = format_json(img_path, boxes)
            json.dump(j, f)
            f.write('\n')
