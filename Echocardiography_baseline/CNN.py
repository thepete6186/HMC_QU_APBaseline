import numpy as np
import os

x_train = np.load(
    os.path.join(
        os.path.join(os.getcwd(), "DataSplits"), "x_train_2CH.npy"
    ) )

print("x_train shape: ", x_train.shape)