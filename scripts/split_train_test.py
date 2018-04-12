import os
import random
from shutil import copyfile
from tqdm import tqdm

def mkdir_if_not_exist(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

SRC_DIR='/home/yuchen/School/ift6135/celeba_data'
DST_DIR='./data'
train_dir = os.path.join(DST_DIR, 'train')
test_dir = os.path.join(DST_DIR, 'test')

mkdir_if_not_exist(DST_DIR)
mkdir_if_not_exist(train_dir)
mkdir_if_not_exist(test_dir)

imgs_name = os.listdir(SRC_DIR)
random.shuffle(imgs_name)
split =int(0.9 * len(imgs_name))
train_imgs_name = imgs_name[0:split]
test_imgs_name = imgs_name[split:]

copy_paths = []
for name in train_imgs_name:
    src_path = os.path.join(SRC_DIR, name)
    dst_path = os.path.join(train_dir, name)
    copy_paths.append((src_path, dst_path))

for name in test_imgs_name:
    src_path = os.path.join(SRC_DIR, name)
    dst_path = os.path.join(test_dir, name)
    copy_paths.append((src_path, dst_path))

for src, dst in tqdm(copy_paths):
    copyfile(src, dst)
