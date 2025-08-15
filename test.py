# Test deep neural network model
import os
import time

import cv2
from keras.optimizers import *
from tqdm import *

from losses import *
from models import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Test {} dataset, using {} model with {} loss ...".format(DATASET_NAME, NETWORK_MODEL, LOSS_FUNCTION))

img_in_dir = os.path.join("./datasets", DATASET_NAME, "test")
print("Image input directory: {}".format(img_in_dir))

img_out_dir = os.path.join(img_in_dir, "Binarized")
if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
print("Image output directory: {}".format(img_out_dir))

model = build_model(NETWORK_MODEL.lower(),
                    num_classes=1,
                    input_height=TILE_SIZE,
                    input_width=TILE_SIZE,
                    num_filters=NUM_FILTERS)

model.compile(optimizer=Adam(), loss=build_loss(LOSS_FUNCTION.lower()), metrics=["accuracy"])

model_name = "..."
print("Loading weights: {}".format(model_name))
model.load_weights(model_name)

img_list = sorted(os.listdir(img_in_dir))
start_time = time.time()
for idx in trange(len(img_list)):
    if os.path.isdir(os.path.join(img_in_dir, img_list[idx])):
        continue

    # print("Now processing the image:", os.path.join(img_in_dir, img_list[idx]))
    fname, fext = os.path.splitext(img_list[idx])

    img = np.asarray(cv2.imread(os.path.join(img_in_dir, img_list[idx]), cv2.IMREAD_GRAYSCALE)) / 255.0
    locations, patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    patches = np.expand_dims(patches, axis=-1)  # patches = np.squeeze(np.expand_dims(patches, axis=-1))
    predictions = model.predict(patches, batch_size=BATCH_SIZE, verbose=2)
    estimated = stitch_together(locations, predictions[:, :, :, 0], tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)
    result = np.where(estimated >= 0.5, 1, 0)
    cv2.imwrite(os.path.join(img_out_dir, fname + "-XW_" + NETWORK_MODEL + "_" + LOSS_FUNCTION + ".png"), result * 255)

print("Total running time: %f sec." % (time.time() - start_time))
print("Finished!")
