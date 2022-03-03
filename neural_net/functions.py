from tensorflow.python.keras.backend import set_session
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import os


def setUpGPU():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            set_session(tf.compat.v1.InteractiveSession(config=config))
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            tf.compat.v1.disable_eager_execution()
        except RuntimeError as e:
            print(e)


# splits dataset to images and reshapes them to 106x106x1
def organizeData(df, num):
    data = []
    for i in range(0, num):
        index = df.shape[1] * i
        image = np.array(df.iloc[range(index, index + df.shape[1]), :])
        image = image.reshape(df.shape[1], df.shape[1], 1)
        data.append(image)

    return data


# returns the one hot encodings corresponding to the labels passed
def oneHotEncode(labels):
    label_encoder = OneHotEncoder(handle_unknown="error", sparse=False)
    return label_encoder.fit_transform(labels)

