import os
import numpy as np
from scipy.optimize.optimize import fmin
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from PIL import Image
from scipy.optimize import \
    fmin_l_bfgs_b  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
import os
import multiprocessing as mp

# dev = tf.config.list_physical_devices('GPU')[0]
# config = tf.config.experimental.set_memory_growth(dev, True)

random.seed(1492)
np.random.seed(1492)
tf.compat.v1.set_random_seed(1492)  # Uncomment for TF1.
tf.random.set_seed(1492)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAMPLE = 1
IMG_DIR_PATH = f"/homes/rose142/cs390/CS390-NIP-Final-Project"
CONTENT_IMG_DIR_PATH = f"{IMG_DIR_PATH}/content"
STYLE_IMG_PATH = f"{IMG_DIR_PATH}/style.jpg"
RESULT_IMG_DIR_PATH = f"{IMG_DIR_PATH}/results"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.02  # Alpha weight.
STYLE_WEIGHT = 0.98  # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 1

# =============================<Helper Functions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


def deprocessImage(img):
    img = np.copy(img).reshape((STYLE_IMG_H, STYLE_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img, 'RGB')


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ========================<Loss Function Builder Functions>======================
class Wrapper:
    def __init__(self, content, style, result):
        tf.compat.v1.disable_eager_execution()
        K.set_floatx('float64')
        self.content = content
        self.style = style
        # print("   Building transfer model.")
        self.contentTensor = content
        self.styleTensor = K.variable(self.style)
        # self.genTensor = K.placeholder((1, 250, 250, 3))
        self.genTensor = result
        self.inputTensor = K.concatenate([self.contentTensor, self.styleTensor, self.genTensor], axis=0)
        self.model = vgg19.VGG19(include_top=False, input_tensor=self.inputTensor)
        self.totalLoss = self.constructTotalLoss()
        # self.gradient = self.constructGradient()
        # self.kerasFunction = self.constructKerasFunction()
        # self.runOutput = None

    def styleLoss(self, style: tf.Tensor, gen: tf.Tensor) -> tf.Tensor:
        styleShape = style.shape
        M = (styleShape[0] * styleShape[1]) ** 2
        N = styleShape[2] ** 2
        error = K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4 * N * M)
        return error

    def contentLoss(self, content: tf.Tensor, gen: tf.Tensor) -> tf.Tensor:
        return K.sum(K.square(gen - content))

    def constructTotalLoss(self) -> tf.Tensor:
        outputDict = dict([(layer.name, layer.output) for layer in self.model.layers])
        loss = 0.0
        styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        contentLayerName = "block5_conv2"
        # print("   Calculating content loss.")
        contentLayer = outputDict[contentLayerName]
        contentOutput = contentLayer[0, :, :, :]
        genOutput = contentLayer[2, :, :, :]
        loss += CONTENT_WEIGHT * self.contentLoss(contentOutput, genOutput) / 2
        # print("   Calculating style loss.")
        for layerName in styleLayerNames:
            styleLayer = outputDict[layerName]
            styleOutput = styleLayer[1, :, :, :]
            genOutput = styleLayer[2, :, :, :]
            layerWeight = 1 / self.activeLayers(layerName)
            loss += STYLE_WEIGHT * layerWeight * self.styleLoss(styleOutput, genOutput)
        return loss

    def activeLayers(self, layerName: str) -> int:
        layerCount = 1
        for layer in self.model.layers:
            if layer.name == layerName:
                break
            layerCount += 1
        return layerCount

    def computeTotalLoss(self, gen: np.ndarray):
        gen = gen.reshape((1, CONTENT_IMG_W, CONTENT_IMG_H, 3))
        self.runOutput = self.kerasFunction([gen])
        loss = self.runOutput[0]
        return loss

    def constructGradient(self) -> tf.Tensor:
        grads = K.gradients(self.totalLoss, self.genTensor)
        return grads

    def constructKerasFunction(self):
        outputs = [self.totalLoss, self.gradient]
        return K.function([self.genTensor], outputs)

    def computeGradient(self, gen: np.ndarray) -> np.ndarray:
        grads = self.runOutput[1][0].flatten()
        return grads


# =========================<Pipeline Functions>==================================

def getRawData(content_path, style_path):
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_DIR_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(content_path)
    tImg = cImg.copy()
    sImg = load_img(style_path)
    print("      Images have been loaded.")
    return (
        (cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_temp = img.resize((ih, iw))
        img = np.array(img_temp)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and de-processed images.
'''


def styleTransfer(cData, sData, tData, saveFile):
    print("   VGG19 model loaded.")
    wrapper = Wrapper(cData, sData)
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        x, tLoss, d = fmin_l_bfgs_b(
            wrapper.computeTotalLoss,
            tData,
            fprime=wrapper.computeGradient,
            maxiter=5000,
            maxfun=30
        )
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x)
        img.save(saveFile)  # Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")

# =========================<Main>================================================

def run_model(i, content_path, style_path, result_path):
    print(f"Beginning transfer of content{i}.")
    raw = getRawData(f'{content_path}/content{i}.jpg', style_path)
    cData = preprocessData(raw[0])  # Content image.
    sData = preprocessData(raw[1])  # Style image.
    tData = preprocessData(raw[2])  # Transfer image.
    styleTransfer(cData, sData, tData, f"{result_path}/result{i}.jpg")


def main():
    print("Starting style transfer program.")
    for i in range(len(os.listdir(CONTENT_IMG_DIR_PATH))):
        process = mp.Process(target=run_model, args=[i])
        process.start()
        process.join()
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
