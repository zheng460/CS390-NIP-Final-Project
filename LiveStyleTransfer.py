import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import random
from PIL import Image
from scipy.optimize import (
    fmin_l_bfgs_b,
)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import time
import warnings
import math

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_SET_DIR_PATH = "./data_set"
TRAINING_DATA_DIR_PATH = f"{DATA_SET_DIR_PATH}/training"
TRAINING_CONTENT_DIR_PATH = f"{TRAINING_DATA_DIR_PATH}/content"
TRAINING_STYLE_DIR_PATH = f"{TRAINING_DATA_DIR_PATH}/style"
TESTING_DATA_DIR_PATH = f"{DATA_SET_DIR_PATH}/testing"
TESTING_CONTENT_DIR_PATH = f"{TESTING_DATA_DIR_PATH}/content"
TESTING_STYLE_DIR_PATH = f"{TESTING_DATA_DIR_PATH}/style"
# image sizes
CONTENT_IMG_H = 250
CONTENT_IMG_W = 250

STYLE_IMG_H = 250
STYLE_IMG_W = 250

tf.random.set_seed(1492)

class Encoder(object):
    def __init__(self):
        # initialize the data
        pass

    def train(self, content_image, style_image):
        # train the encoder based on the images
        pass

    def loss(self):
        # loss function
        pass

    def grad(self):
        # calculate the gradient
        pass

    def run(self, model):
        # encode the information
        pass


# =============================<Helper Functions>=================================
# load images
def preprocess_data_set(data_set):
    ((x_training, y_training), (x_testing, y_testing)) = data_set
    image_dimensions = (CONTENT_IMG_H, CONTENT_IMG_W)
    x_training = preprocess_data(x_training, image_dimensions)
    y_training = preprocess_data(y_training, image_dimensions)
    x_testing = preprocess_data(x_testing, image_dimensions)
    y_testing = preprocess_data(y_testing, image_dimensions)
    return ((x_training, y_training), (x_testing, y_testing))


def preprocess_data(data, dimensions):
    return np.array([preprocess_image(image, dimensions) for image in data])


def preprocess_image(image: Image.Image, dimensions):
    img = image
    img = np.array(img.resize(dimensions))
    img = img.astype("float64")
    img = img / 255.0
    return img


def load_images(dir):
    files = os.listdir(dir)
    list_of_images = []
    for filename in files:
        cImg = load_img(
            f"{dir}/{filename}"
        )
        list_of_images.append(cImg)
    return list_of_images


def load_data():
    x_training = load_images(TRAINING_CONTENT_DIR_PATH)
    y_training = load_images(TRAINING_STYLE_DIR_PATH)
    x_testing = load_images(TESTING_CONTENT_DIR_PATH)
    y_testing = load_images(TESTING_STYLE_DIR_PATH)
    return ((x_training, y_training), (x_testing, y_testing))


def get_data_set():
    data_set = load_data()
    return preprocess_data_set(data_set)


# =============================<Helper Functions>=================================


def train():
    pass


def evaluate(model):
    test_images = load_images("./data_set/testing/content")
    for img in test_images:
        result = model.run(img)
        # im = Image.fromarray(result)
        # im.save("./results/result.jpg")


def main():
    # images = load_images("./data_set/training/content")
    images = get_data_set()

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (2, 2), strides=(2, 2), activation='relu', input_shape=(CONTENT_IMG_H, CONTENT_IMG_W, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Reshape(math.sqrt(4096), math.sqrt(4096)))
    model.add(keras.layers.Conv2DTranspose(128, (2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(3, (2, 2), activation='sigmoid', padding='same'))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x=images[0][0], y=images[0][1], epochs=25, batch_size=1)
    model.evaluate(x=images[1][0], y=images[1][1])

    prediction = model.predict(images[1][0])[0]
    prediction = np.round(prediction * 255.0).astype('int8')
    img = Image.fromarray(prediction, mode='RGB')
    img.save(f'{DATA_SET_DIR_PATH}/predicted.jpg')

    prediction = images[1][1][0]
    prediction = np.round(prediction * 255.0).astype('int8')
    img = Image.fromarray(prediction, mode='RGB')
    img.save(f'{DATA_SET_DIR_PATH}/expected.jpg')

    # style = load_img(
    #     "./data_set/training/style/style.jpg",
    #     target_size=(CONTENT_IMG_W, CONTENT_IMG_H),
    # )
    # model = Encoder()
    # for image in images:
    #     model.train(image, style)
    # evaluate(model)


if __name__ == "__main__":
    main()
