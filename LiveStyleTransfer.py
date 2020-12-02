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
from tensorflow.keras.applications import vgg19
import time
import warnings
import math
import lab3

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

# session = tf.compat.v1.keras.backend.get_session()
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)
tf.compat.v1.disable_eager_execution()

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
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    img = np.squeeze(img, axis=0)
    return img


def load_images(dir, filename):
    files = os.listdir(dir)
    list_of_images = []
    for i in range(len(files)):
        cImg = load_img(
            f"{dir}/{filename}{i}.jpg"
        )
        list_of_images.append(cImg)
    return list_of_images


def load_data():
    x_training = load_images(TRAINING_CONTENT_DIR_PATH, 'content')
    y_training = load_images(TRAINING_STYLE_DIR_PATH, 'style')
    x_testing = load_images(TESTING_CONTENT_DIR_PATH, 'content')
    y_testing = load_images(TESTING_STYLE_DIR_PATH, 'style')
    return ((x_training, y_training), (x_testing, y_testing))


def get_data_set():
    data_set = load_data()
    return preprocess_data_set(data_set)


# =============================<Helper Functions>=================================


def build_unet():
    inputs = keras.Input(shape=(CONTENT_IMG_H, CONTENT_IMG_W, 3))

    conv1 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    leaky1 = keras.layers.LeakyReLU()(conv1)
    batch1 = keras.layers.BatchNormalization()(leaky1)
    conv2 = keras.layers.Conv2D(64, (2, 2))(batch1)
    leaky2 = keras.layers.LeakyReLU()(conv2)
    batch2 = keras.layers.BatchNormalization()(leaky2)
    conv3 = keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same')(batch2)
    leaky3 = keras.layers.LeakyReLU()(conv3)
    batch3 = keras.layers.BatchNormalization()(leaky3)
    conv4 = keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same')(batch3)
    leaky4 = keras.layers.LeakyReLU()(conv4)
    batch4 = keras.layers.BatchNormalization()(leaky4)
    conv5 = keras.layers.Conv2D(64, (2, 2))(batch4)
    leaky5 = keras.layers.LeakyReLU()(conv5)
    batch5 = keras.layers.BatchNormalization()(leaky5)
    conv6 = keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same')(batch5)
    leaky6 = keras.layers.LeakyReLU()(conv6)
    batch6 = keras.layers.BatchNormalization()(leaky6)

    flat = keras.layers.Flatten()(batch6)
    dense1 = keras.layers.Dense(15000)(flat)
    leaky_dense1 = keras.layers.LeakyReLU()(dense1)
    batch_dense1 = keras.layers.BatchNormalization()(leaky_dense1)
    dense2 = keras.layers.Dense(5120)(batch_dense1)
    leaky_dense2 = keras.layers.LeakyReLU()(flat)
    batch_dense2 = keras.layers.BatchNormalization()(leaky_dense2)
    dense3 = keras.layers.Dense(15 * 15 * 64)(batch_dense2)
    leaky_dense3 = keras.layers.LeakyReLU()(dense3)
    batch_dense3 = keras.layers.BatchNormalization()(leaky_dense3)
    reshape = keras.layers.Reshape((15, 15, 64))(batch_dense3)

    merge1 = keras.layers.concatenate([reshape, batch6])
    conv_t1 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(merge1)
    leaky_t1 = keras.layers.LeakyReLU()(conv_t1)
    batch_t1 = keras.layers.BatchNormalization()(leaky_t1)
    merge2 = keras.layers.concatenate([batch_t1, batch5])
    conv_t2 = keras.layers.Conv2DTranspose(64, (2, 2))(merge2)
    leaky_t2 = keras.layers.LeakyReLU()(conv_t2)
    batch_t2 = keras.layers.BatchNormalization()(leaky_t2)
    merge3 = keras.layers.concatenate([batch_t2, batch4])
    conv_t3 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(merge3)
    leaky_t3 = keras.layers.LeakyReLU()(conv_t3)
    batch_t3 = keras.layers.BatchNormalization()(leaky_t3)
    merge4 = keras.layers.concatenate([batch_t3, batch3])
    conv_t4 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(merge4)
    leaky_t4 = keras.layers.LeakyReLU()(conv_t4)
    batch_t4 = keras.layers.BatchNormalization()(leaky_t4)
    merge5 = keras.layers.concatenate([batch_t4, batch2])
    conv_t5 = keras.layers.Conv2DTranspose(64, (2, 2))(merge5)
    leaky_t5 = keras.layers.LeakyReLU()(conv_t5)
    batch_t5 = keras.layers.BatchNormalization()(leaky_t5)
    merge6 = keras.layers.concatenate([batch_t5, batch1])
    conv_t6 = keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(merge6)
    leaky_t6 = keras.layers.LeakyReLU()(conv_t6)
    batch_t6 = keras.layers.BatchNormalization()(leaky_t6)

    merge_output1 = keras.layers.concatenate([batch_t6, inputs])
    outputs1 = keras.layers.Conv2DTranspose(3, (2, 2), padding='same')(merge_output1)
    leaky_output1 = keras.layers.LeakyReLU()(outputs1)
    batch_output1 = keras.layers.BatchNormalization()(leaky_output1)

    # merge_output2 = keras.layers.concatenate([batch_output1, inputs])
    # outputs2 = keras.layers.Conv2DTranspose(3, (1, 1), activation='sigmoid', padding='same')(merge_output2)

    model = keras.Model(inputs=inputs, outputs=batch_output1, name='LiveStyleTransfer-UNet')
    model.summary()
    model.compile(optimizer='adam', loss=calculate_loss)
    return model


def deprocessImage(img):
    img = np.copy(img).reshape((STYLE_IMG_H, STYLE_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img, 'RGB')

def save_image(img, location):
    deprocessImage(img).save(location)


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


S_DATA = np.expand_dims(preprocess_image(load_img(f"{DATA_SET_DIR_PATH}/style_source.jpg"), (STYLE_IMG_H, STYLE_IMG_W)), axis=0)

def calculate_loss(expected, predicted):
    expected = tf.cast(expected, tf.float64)
    predicted = tf.cast(predicted, tf.float64)
    # predicted = K.clip(predicted, 0, 255)
    wrapper = lab3.Wrapper(expected, S_DATA, predicted)
    return wrapper.constructTotalLoss()

def main():
    images = get_data_set()
    expected = images[1][1][0]
    save_image(expected, f'{DATA_SET_DIR_PATH}/expected.jpg')
    model = build_unet()

    model.fit(x=images[0][0], y=images[0][1], epochs=15, batch_size=1)
    model.evaluate(x=images[1][0], y=images[1][1])

    prediction = model.predict(np.array([images[1][0][0]]))[0]
    save_image(prediction, f'{DATA_SET_DIR_PATH}/predicted.jpg')

    model.save('/homes/rose142/scratch/model_result')


if __name__ == "__main__":
    main()
