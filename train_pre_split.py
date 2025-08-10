# Train deep neural network model
# Training and validation datasets are already pre-split

import pickle
import random
from datetime import datetime

from keras.callbacks import *
from keras.preprocessing.image import *

from global_variables import *
from losses import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for _ in range(1):
    K.clear_session()

    data_root = os.path.join("./datasets", DATASET_NAME, "output_train_0.8_val_0.2_seed_3376")
    print("Training {}, using {} model with {} loss ...".format(DATASET_NAME, NETWORK_MODEL, LOSS_FUNCTION))

    train_val_split_str = data_root[data_root.rindex("/") + 8:data_root.rindex("/") + 25]  # extract substring: train_0.8_val_0.2

    data_train_dir = os.path.join(data_root, "train")
    data_val_dir = os.path.join(data_root, "val")

    random.seed()
    seed1 = round(random.random() * 10000)
    seed2 = round(random.random() * 10000)
    print("Random seed1: {}, and random seed2: {}".format(seed1, seed2))

    train_img_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_msk_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_img_generator = train_img_datagen.flow_from_directory(
        data_train_dir,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["images"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed1,
    )

    train_msk_generator = train_msk_datagen.flow_from_directory(
        data_train_dir,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["labels"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed1,
    )

    val_img_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_msk_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    val_img_generator = val_img_datagen.flow_from_directory(
        data_val_dir,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["images"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed2,
    )

    val_msk_generator = val_msk_datagen.flow_from_directory(
        data_val_dir,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["labels"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed2,
    )

    train_generator = zip(train_img_generator, train_msk_generator)
    val_generator = zip(val_img_generator, val_msk_generator)

    model = build_model(NETWORK_MODEL.lower(),
                        num_classes=1,
                        input_height=TILE_SIZE,
                        input_width=TILE_SIZE,
                        num_filters=NUM_FILTERS)

    model.compile(optimizer="Adam", loss=build_loss(LOSS_FUNCTION.lower()), metrics=["accuracy"])

    # model.load_weights("...")
    # model.summary()

    model_weights_root = "./weights-" + DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() + "-" + LOSS_FUNCTION.lower() + "-" + train_val_split_str + "-" + str(
        datetime.timestamp(datetime.now()))
    if not os.path.exists(model_weights_root):
        os.makedirs(model_weights_root)

    check_point = ModelCheckpoint(os.path.join(model_weights_root,
                                               DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() + "-" + LOSS_FUNCTION.lower() + "-" + train_val_split_str +
                                               "-ps_" + str(TILE_SIZE) + "x" + str(TILE_SIZE) +
                                               "-ch_" + str(NUM_FILTERS) +
                                               "-bs_" + str(BATCH_SIZE) +
                                               "-val_loss_{val_loss}-val_accuracy_{val_accuracy}.hdf5"),
                                  monitor="val_loss",
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode="auto")

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  patience=10,
                                  verbose=1,
                                  mode="auto")

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=15,
                               verbose=1,
                               mode="auto")

    print("Now start training the network model...")
    history = model.fit_generator(train_generator,
                                  epochs=NUM_EPOCHS,
                                  verbose=2,
                                  steps_per_epoch=len(train_img_generator),
                                  validation_data=val_generator,
                                  validation_steps=len(val_img_generator),
                                  callbacks=[check_point, reduce_lr, early_stop])

    with open(os.path.join(model_weights_root,
                           DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() + "-" + LOSS_FUNCTION.lower() + "-" + train_val_split_str +
                           "-ps_" + str(TILE_SIZE) + "x" + str(TILE_SIZE) +
                           "-ch_" + str(NUM_FILTERS) +
                           "-bs_" + str(BATCH_SIZE) +
                           "-" + str(datetime.timestamp(datetime.now())) + ".history"), "wb") as df:
        pickle.dump(history.history, df)

    df.close()

print("Finished!")
