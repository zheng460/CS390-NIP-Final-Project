import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import random
from PIL import Image
from skimage import transform,io
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
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

class Encoder(object):
    def __init__(self):
        #initilize the data
        pass

    def train(self,content_image,style_image):
        #train the encoder based on the images
        pass

    def loss(self):
        #loss fucntion
        pass
    def grad(self):
        #calculate the gradient
        pass
    def run(self,model):
        #encode the information
        pass

#=============================<Helper Fuctions>=================================
#load images
def preprocess_data_set(data_set):
    (
        (x_training, y_training),
        (x_testing, y_testing)
    ) = data_set
    image_dimensions = (CONTENT_IMG_H, CONTENT_IMG_W)
    x_training = preprocess_data((x_training, image_dimensions))
    y_training = preprocess_data((y_training, image_dimensions))
    x_testing = preprocess_data((x_testing, image_dimensions))
    y_testing = preprocess_data((y_testing, image_dimensions))
    return ((x_training, y_training), (x_testing, y_testing))

def preprocess_data(data, dimensions):
    return [ preprocess_image(image, dimensions) for image in data ]

def preprocess_image(image, dimensions):
    img = image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_temp = img.resize(dimensions)
        img = np.array(img_temp)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    return img

def load_images(dir):
    files = os.listdir(dir)
    list_of_images = []
    for filename in files:
        cImg = load_img(dir+'/'+filename, target_size=(CONTENT_IMG_W, CONTENT_IMG_H))
        list_of_images.append(cImg)
        print("load image",filename)
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

#=============================<Helper Fuctions>=================================

def preprocess_data(raw_data):
    pass

def train():
    pass

def evaluate(model):
    test_images = load_images("./data_set/testing/content")
    for img in test_images:
        result = model.run(img)
        #im = Image.fromarray(result)
        #im.save("./results/result.jpg")


def main():
    images = load_images("./data_set/training/content")
    style = load_img("./data_set/training/sytle/style.jpg", target_size=(CONTENT_IMG_W, CONTENT_IMG_H))
    model = Encoder()
    for image in images:
        model.train(image,style)
    evaluate(model)


if __name__ == "__main__":
    main()
