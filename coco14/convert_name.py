import glob
import os
import pdb
# from tqdm import tqdm

file_dir = "/mnt/shyoon4tb/RIB/Dataset/coco_2014/JPEGImages/*.jpg"

files = glob.glob(file_dir)

print(len(files))

for file in files:

    if 'train2014' in os.path.basename(file):
        tgt_name = os.path.join(os.path.dirname(file),os.path.basename(file)[15:])
        os.rename(file,tgt_name)
    elif 'val2014' in os.path.basename(file):
        tgt_name = os.path.join(os.path.dirname(file),os.path.basename(file)[13:])
        os.rename(file,tgt_name)
        # pdb.set_trace()