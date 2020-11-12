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
def load_images(dir):
    files = os.listdir(dir)
    list_of_images = []
    for filename in files:
        cImg = load_img(filename, target_size=(CONTENT_IMG_W, CONTENT_IMG_H))
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

def evaluate(model, style):
    test_images = load_images("./data_set/testing/content")
    for img in test_images:
        result = model.run(img,style)
        im = Image.fromarray(result)
        im.save("./results/result.jpg")


def main():
    images = load_images("./data_set/training/content")
    style = load_img("./data_set/training/sytle/style.jpg", target_size=(CONTENT_IMG_W, CONTENT_IMG_H))
    model = Encoder()
    for image in images:
        model.train(image,style)
    evaluate(model)


if __name__ == "__main__":
    main()
