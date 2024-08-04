import glob
import os
import pdb
import numpy as np

file_dir = "/mnt/shyoon4tb/RIB/Dataset/coco_2014/JPEGImages/*.jpg"


files = glob.glob(file_dir)

cls_dict = np.load("/mnt/shyoon3/wsss_ysh_coco/coco/cls_labels.npy",allow_pickle=True).item()
new_cls_dict = {}

print(len(files))

for file in files:
    print(file)
    # print(cls_dict[file])

    