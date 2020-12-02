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

DATA_SET_DIR_PATH = "./data_set"
TRAINING_DATA_DIR_PATH = f"{DATA_SET_DIR_PATH}/training"
TRAINING_CONTENT_DIR_PATH = f"{TRAINING_DATA_DIR_PATH}/content"
TRAINING_STYLE_DIR_PATH = f"{TRAINING_DATA_DIR_PATH}/style"
TESTING_DATA_DIR_PATH = f"{DATA_SET_DIR_PATH}/testing"
TESTING_CONTENT_DIR_PATH = f"{TESTING_DATA_DIR_PATH}/content"
TESTING_STYLE_DIR_PATH = f"{TESTING_DATA_DIR_PATH}/style"
# image sizes
CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500
os.environ["CUDA_VISIBLE_DEVICES"]="4"

class Encoder(object):
    def __init__(self):
        # initilize the data
        pass

    def train(self, content_image, style_image):
        # train the encoder based on the images
        pass

    def loss(self):
        # loss fucntion
        pass

    def grad(self):
        # calculate the gradient
        pass

    def run(self, model):
        # encode the information
        pass


# =============================<Helper Fuctions>=================================
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


# =============================<Helper Fuctions>=================================


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
    model.add(keras.Input(shape=(CONTENT_IMG_H, CONTENT_IMG_W, 3)))
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", strides=(2,2), padding="same"))
    model.add(keras.layers.Conv2D(64, (2,2), activation="relu", strides=(2,2), padding="same"))
    model.add(keras.layers.Conv2D(64, (2, 2), activation="relu", strides=(2,2), padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024,activation="relu"))
    model.add(keras.layers.Dense(125*125*64,activation="relu"))
    model.add(keras.layers.Reshape((125,125,64)))
    model.add(keras.layers.Conv2DTranspose(64, (2,2), activation="relu", strides=(2,2), padding="same"))
    model.add(keras.layers.Conv2DTranspose(32, (3,3), activation="relu", strides=(2,2), padding="same"))
    model.add(keras.layers.Conv2DTranspose(3,3, activation="sigmoid", padding="same"))
    #model.add(keras.layers.Dense(16, activation="relu"))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    for i in range(100):
        with tf.device('/gpu:1'):
            model.fit(x=images[0][0], y=images[0][1],batch_size= 3,epochs=10)
        timage = load_img("./data_set/testing/content/content0.jpg")
        result = preprocess_image(timage,(CONTENT_IMG_H, CONTENT_IMG_W))
        result = timage
        result = np.array(result)
        result= result.astype("float64")
        result = result.reshape(1,500,500,3)
        rimg = model.predict(result)[0]
        print(rimg)
        rimg = rimg*255
        rimg = rimg.astype('int')

        print(rimg)
        im = Image.fromarray(rimg,mode = 'RGB')
        im.save(f"result{i}.jpg")
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
