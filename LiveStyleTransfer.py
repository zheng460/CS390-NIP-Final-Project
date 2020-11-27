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

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_SET_DIR_PATH = "/homes/rose142/cs390/CS390-NIP-Final-Project/data_set"
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


def build_unet():
    inputs = keras.Input(shape=(CONTENT_IMG_H, CONTENT_IMG_W, 3))

    conv1 = keras.layers.Conv2D(128, (2, 2), strides=(2, 2))(inputs)
    leaky1 = keras.layers.LeakyReLU()(conv1)
    batch1 = keras.layers.BatchNormalization()(leaky1)
    conv2 = keras.layers.Conv2D(128, (2, 2), strides=(2, 2))(batch1)
    leaky2 = keras.layers.LeakyReLU()(conv2)
    batch2 = keras.layers.BatchNormalization()(leaky2)
    conv3 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2))(batch2)
    leaky3 = keras.layers.LeakyReLU()(conv3)
    batch3 = keras.layers.BatchNormalization()(leaky3)
    conv4 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2))(batch3)
    leaky4 = keras.layers.LeakyReLU()(conv4)
    batch4 = keras.layers.BatchNormalization()(leaky4)
    conv5 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2))(batch4)
    leaky5 = keras.layers.LeakyReLU()(conv5)
    batch5 = keras.layers.BatchNormalization()(leaky5)

    flat = keras.layers.Flatten()(batch5)
    dense = keras.layers.Dense(4608)(flat)
    leaky_dense = keras.layers.LeakyReLU()(dense)
    batch_dense = keras.layers.BatchNormalization()(leaky_dense)
    reshape = keras.layers.Reshape((6, 6, 128))(batch_dense)

    merge1 = keras.layers.concatenate([reshape, batch5])
    conv_t1 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2))(merge1)
    leaky_t1 = keras.layers.LeakyReLU()(conv_t1)
    batch_t1 = keras.layers.BatchNormalization()(leaky_t1)

    merge2 = keras.layers.concatenate([batch_t1, batch4])
    conv_t2 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2))(merge2)
    leaky_t2 = keras.layers.LeakyReLU()(conv_t2)
    batch_t2 = keras.layers.BatchNormalization()(leaky_t2)

    merge3 = keras.layers.concatenate([batch_t2, batch3])
    conv_t3 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2))(merge3)
    leaky_t3 = keras.layers.LeakyReLU()(conv_t3)
    batch_t3 = keras.layers.BatchNormalization()(leaky_t3)

    merge4 = keras.layers.concatenate([batch_t3, batch2])
    conv_t4 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(merge4)
    leaky_t4 = keras.layers.LeakyReLU()(conv_t4)
    batch_t4 = keras.layers.BatchNormalization()(leaky_t4)

    conv_t5 = keras.layers.Conv2DTranspose(128, (2, 2))(batch_t4)
    leaky_t5 = keras.layers.LeakyReLU()(conv_t5)
    batch_t5 = keras.layers.BatchNormalization()(leaky_t5)

    merge5 = keras.layers.concatenate([batch_t5, batch1])
    conv_t6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(merge5)
    leaky_t6 = keras.layers.LeakyReLU()(conv_t6)
    batch_t6 = keras.layers.BatchNormalization()(leaky_t6)

    outputs = keras.layers.Conv2DTranspose(3, (2, 2), activation='sigmoid', padding='same')(batch_t6)

    model = keras.Model(inputs=inputs, outputs=outputs, name='LiveStyleTransfer-UNet')
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def evaluate(model):
    test_images = load_images("./data_set/testing/content")
    for img in test_images:
        result = model.run(img)
        # im = Image.fromarray(result)
        # im.save("./results/result.jpg")


def main():
    # images = load_images("./data_set/training/content")
    images = get_data_set()
    print(images[1][0][0].shape)
    model = build_unet()

    model.fit(x=images[0][0], y=images[0][1], epochs=25)
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
