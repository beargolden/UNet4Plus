# Prepare dataset for training (with auto train-val split)

import os
import random

import cv2
import splitfolders

from utils import *

validation_split = 0.2

random.seed()
seed = round(random.random() * 10000)

data_root = os.path.join("./datasets", DATASET_NAME)
print("Data root directory: {}, and image patch size: {} x {}".format(data_root, TILE_SIZE, TILE_SIZE))

data_train_path = os.path.join(data_root, "train")
if not os.path.exists(data_train_path):
    os.makedirs(os.path.join(data_train_path, "images"))
    os.makedirs(os.path.join(data_train_path, "labels"))

img_list = sorted(os.listdir(data_root))

for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(data_root, img_list[idx])):
        continue

    print("Now processing image:", os.path.join(data_root, img_list[idx]))
    fname, fext = os.path.splitext(img_list[idx])
    img = cv2.imread(os.path.join(data_root, img_list[idx]), cv2.IMREAD_COLOR)
    msk = cv2.imread(os.path.join(data_root, "GT", fname + "_GT.bmp"), cv2.IMREAD_GRAYSCALE)

    # extract the patches from the original document images and the corresponding ground truths
    img_patch_locations, img_patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    msk_patch_locations, msk_patches = get_patches(msk, TILE_SIZE, TILE_SIZE)

    print("\t{} patches extracted in {}".format(len(img_patches), img_list[idx]))
    for idy in range(len(img_patches)):
        cv2.imwrite(os.path.join(data_train_path, "images", fname + "_" + str(idy) + ".bmp"), img_patches[idy])
        cv2.imwrite(os.path.join(data_train_path, "labels", fname + "_" + str(idy) + ".bmp"), msk_patches[idy])

if validation_split:
    print("Prepare training ({:.1f}) and validation ({:.1f}) subsets with random seed1: {}".format(1 - validation_split, validation_split, seed))
    splitfolders.ratio(data_train_path,
                       output=os.path.join(data_root,
                                           DATASET_NAME.lower() + "_train_{:.1f}_val_{:.1f}_seed_{}".format(1 - validation_split, validation_split,
                                                                                                            seed)),
                       seed=seed,
                       ratio=(1 - validation_split, validation_split),
                       group_prefix=None,
                       move=False)

print("Finished!")
