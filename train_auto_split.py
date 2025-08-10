# Train deep neural network model
# Training and validation datasets will be automatically split by ImageDataGenerator

import pickle
import random
from datetime import datetime

from keras.callbacks import *
from keras.preprocessing.image import *

from global_variables import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for _ in range(1):
    K.clear_session()

    data_root = os.path.join("./datasets", DATASET_NAME)
    print("Training dataset: {}, using network model: {}".format(DATASET_NAME, NETWORK_MODEL))

    data_train_path = os.path.join(data_root, "train")

    random.seed()
    seed = round(random.random() * 10000)
    print("Random seed1: {}".format(seed))

    img_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    msk_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_img_generator = img_datagen.flow_from_directory(
        data_train_path,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["images"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="training",
    )

    train_msk_generator = msk_datagen.flow_from_directory(
        data_train_path,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["labels"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="training",
    )

    valid_img_generator = img_datagen.flow_from_directory(
        data_train_path,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["images"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="validation",
    )

    valid_msk_generator = msk_datagen.flow_from_directory(
        data_train_path,
        target_size=(TILE_SIZE, TILE_SIZE),
        color_mode="grayscale",
        classes=["labels"],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="validation",
    )

    train_generator = zip(train_img_generator, train_msk_generator)
    valid_generator = zip(valid_img_generator, valid_msk_generator)

    model = build_model(NETWORK_MODEL,
                        num_classes=1,
                        input_height=TILE_SIZE,
                        input_width=TILE_SIZE,
                        num_filters=NUM_FILTERS)

    model.compile(optimizer=Adam(), loss=["binary_crossentropy"], metrics=["accuracy"])

    # model.load_weights("...")
    # model.summary()

    model_weights_root = "./weights-" + DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() + "-" + str(datetime.timestamp(datetime.now()))
    if not os.path.exists(model_weights_root):
        os.makedirs(model_weights_root)

    check_point = ModelCheckpoint(os.path.join(model_weights_root, DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() +
                                               "-ps_" + str(TILE_SIZE) + "x" + str(TILE_SIZE) +
                                               "-ch_" + str(NUM_FILTERS) +
                                               "-bs_" + str(BATCH_SIZE) +
                                               "-val_loss_{val_loss:.5f}-val_accuracy_{val_accuracy:.5f}.hdf5"),
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
                                  validation_data=valid_generator,
                                  validation_steps=len(valid_img_generator),
                                  callbacks=[check_point, reduce_lr, early_stop])

    with open(os.path.join(model_weights_root, DATASET_NAME.lower() + "-" + NETWORK_MODEL.lower() +
                                               "-ps_" + str(TILE_SIZE) + "x" + str(TILE_SIZE) +
                                               "-ch_" + str(NUM_FILTERS) +
                                               "-bs_" + str(BATCH_SIZE) +
                                               "-" + str(datetime.timestamp(datetime.now())) + ".history"), "wb") as df:
        pickle.dump(history.history, df)

    df.close()

print("Finished!")
