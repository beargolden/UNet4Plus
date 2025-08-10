# Prepare dataset for training and validation

import os
import random

import cv2

from utils import *

RANDOM_SEED = 2025

data_root = os.path.join("./datasets", DATASET_NAME)
print("Data root directory: {}, and image patch size: {} x {}".format(data_root, TILE_SIZE, TILE_SIZE))

data_train_path = os.path.join(data_root, "train")
if not os.path.exists(data_train_path):
    os.makedirs(os.path.join(data_train_path, "images"))
    os.makedirs(os.path.join(data_train_path, "labels"))

data_valid_path = os.path.join(data_root, "valid")
if not os.path.exists(data_valid_path):
    os.makedirs(os.path.join(data_valid_path, "images"))
    os.makedirs(os.path.join(data_valid_path, "labels"))

img_list = sorted(os.listdir(data_root))

total_img_patches, total_msk_patches = [], []

for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(data_root, img_list[idx])):
        continue

    print("Now processing image:", os.path.join(data_root, img_list[idx]))
    fname, fext = os.path.splitext(img_list[idx])
    img = cv2.imread(os.path.join(data_root, img_list[idx]), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(data_root, "GT", fname + "_GT.bmp"), cv2.IMREAD_GRAYSCALE)

    # extract the patches from the original document images and the corresponding ground truths
    img_patch_locations, img_patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    msk_patch_locations, msk_patches = get_patches(msk, TILE_SIZE, TILE_SIZE)

    print("\t{} patches extracted...".format(len(img_patches)))
    for idy in range(len(img_patches)):
        total_img_patches.append(img_patches[idy])
        total_msk_patches.append(msk_patches[idy])

random.seed(RANDOM_SEED)  # !important
random.shuffle(total_img_patches)
random.seed(RANDOM_SEED)  # !important
random.shuffle(total_msk_patches)

train_valid_split = round(0.8 * len(total_img_patches))
print("Number of training patches: {}, and number of validation patches: {}".format(train_valid_split, len(total_img_patches) - train_valid_split))

train_img_patches = total_img_patches[:train_valid_split]
valid_img_patches = total_img_patches[train_valid_split:]
train_msk_patches = total_msk_patches[:train_valid_split]
valid_msk_patches = total_msk_patches[train_valid_split:]

for idz in range(len(train_img_patches)):
    cv2.imwrite(os.path.join(data_train_path, "images", DATASET_NAME.lower() + "_" + str(idz) + ".bmp"), train_img_patches[idz])
    cv2.imwrite(os.path.join(data_train_path, "labels", DATASET_NAME.lower() + "_" + str(idz) + ".bmp"), train_msk_patches[idz])

for idz in range(len(valid_img_patches)):
    cv2.imwrite(os.path.join(data_valid_path, "images", DATASET_NAME.lower() + "_" + str(idz) + ".bmp"), valid_img_patches[idz])
    cv2.imwrite(os.path.join(data_valid_path, "labels", DATASET_NAME.lower() + "_" + str(idz) + ".bmp"), valid_msk_patches[idz])

print("Finished!")
