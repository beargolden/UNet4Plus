# Visualize training history

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# load history file
his_dir = "~/UNet4Plus/weights-bickleydiary-unet4plus-1629195384.303941"
names = os.listdir(his_dir)
for name in names:
    if name.endswith(".history"):
        df = open(os.path.join(his_dir, name), "rb")
        history = pickle.load(df)
        df.close()

        # # list all data in history
        # print(history.keys())
        #
        # # summarize history for accuracy
        # plt.plot(history["accuracy"])
        # plt.plot(history["val_accuracy"])
        # plt.title("model accuracy")
        # plt.ylabel("accuracy")
        # plt.xlabel("epoch")
        # plt.legend(["train", "test"], loc="lower right")
        # plt.show()

        # summarize history for loss
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.plot(np.argmin(history["val_loss"]), min(history["val_loss"]), "xk")
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test", "best model"], loc="upper right")
        # plt.show()

        print("name:", name)
        print("val_loss:", format(min(history["val_loss"]), ".5f"))
        print("val_accuracy:", format(history["val_accuracy"][np.argmin(history["val_loss"])], ".5f"))

        print("val_loss_{}-val_accuracy_{}".format(format(min(history["val_loss"]), ".5f"),
                                                   format(history["val_accuracy"][np.argmin(history["val_loss"])], ".5f")))
