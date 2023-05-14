from __future__ import print_function, division
from cgi import test
from imghdr import tests
import sys
import os
import time
import tensorflow as tf

from keras.layers import Input
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import BinaryCrossentropy
# from tkeras.layers.activations import sigmoid
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
from keras.metrics import BinaryAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

import numpy as np
import math
import logging
from datetime import date, datetime
import random

entireImgDir = '/bigdata/ethan/drv_prediction_entire_hyper_img_no_padding'

# Logging
progName = sys.argv[0].split('/')[-1]
lowResName = sys.argv[1]
today = str(date.today())
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
logFileName = progName[0:-3] + "-" + today + '-' + current_time + ".log"
if not os.path.exists('./log'):
    os.mkdir('./log')
logging.basicConfig(filename='./log/' + logFileName, level=logging.DEBUG)
logging.info('Start')
logging.info('')

INPUT_SIZE = 64  # size of input box to the model
# This is pixel height, make sure height and width are the same
CROPPED_ROWS = INPUT_SIZE
CROPPED_COLS = INPUT_SIZE  # This is pixel width
COARSE_SIZE = 32
HI_RE_SIZE = 4
PAD_SIZE = int((INPUT_SIZE - HI_RE_SIZE) / 2)
NUM_FEATURES = 7

designSizes = {
    '9t10' : (1337, 1433, 39),
    '9t1' : (98, 97, 39),
    '9t2' : (581, 392, 39),
    '9t3' : (130, 130, 39),
    '9t4' : (534, 517, 23),
    '9t5' : (302, 302, 23),
    '9t6' : (905, 883, 39),
    '9t7' : (1053, 1011, 39),
    '9t8' : (1202, 1138, 39),
    '9t9' : (1337, 1433, 39)
}
trainSet = []
testSet = []

# data paths
designs = ['9t1', '9t2', '9t3', '9t6', '9t7', '9t8', '9t9', '9t10']
entireImgs = {}

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

for design in designs:
    entireImg = np.load(os.path.join(entireImgDir, '%s.npy'%design))
    entireImg = np.pad(entireImg, ((PAD_SIZE,PAD_SIZE), (PAD_SIZE,PAD_SIZE), (0,0)), 'edge')
    entireImgs[design] = np.zeros((entireImg.shape[0], entireImg.shape[1], NUM_FEATURES+1))
    entireImgs[design][:,:,:6] = entireImg[:,:,:6]
    grImg = NormalizeData(np.sum(np.load('/bigdata/ethan/predicted_congestion_maps/predicted_congestion_map_19test%s.npy'%design[2:]), axis=2))
    grImg = np.pad(grImg, ((PAD_SIZE,PAD_SIZE), (PAD_SIZE,PAD_SIZE)), 'edge')
    entireImgs[design][:,:,6] = grImg
    entireImgs[design][:,:,7] = entireImg[:,:,6]
    

logging.basicConfig(format='%(asctime)s %(message)s')
def getHiResTileByCoarseTile(design, idx):
    numCoarseRows = designSizes[design][0] // COARSE_SIZE
    if designSizes[design][0] % COARSE_SIZE != 0:
        numCoarseRows += 1
    numCoarseCols = designSizes[design][1] // COARSE_SIZE
    if designSizes[design][1] % COARSE_SIZE != 0:
        numCoarseCols += 1
    numHiResRows = designSizes[design][0] // HI_RE_SIZE
    if designSizes[design][0] % HI_RE_SIZE != 0:
        numHiResRows += 1
    numHiResCols = designSizes[design][1] // HI_RE_SIZE
    if designSizes[design][1] % HI_RE_SIZE != 0:
        numHiResCols += 1

    rowIdxCoarse = idx // numCoarseCols
    colIdxCoarse = idx % numCoarseCols
    resRatio = COARSE_SIZE // HI_RE_SIZE
    if rowIdxCoarse == numCoarseRows - 1:
        rowIdxHiRes = numHiResRows - resRatio
    else:
        rowIdxHiRes = rowIdxCoarse * resRatio

    if colIdxCoarse == numCoarseCols - 1:
        colIdxHiRes = numHiResCols - resRatio
    else:
        colIdxHiRes = colIdxCoarse * resRatio

    startIdx = rowIdxHiRes * numHiResCols + colIdxHiRes
    tilesHiRes = []
    for i in range(resRatio):
        for j in range(resRatio):
            tilesHiRes.append(startIdx + i * numHiResCols + j)
    return tilesHiRes



def getImgByIdx(entireImg, idx):
    imgsPerRow = (entireImg.shape[0]-INPUT_SIZE) // HI_RE_SIZE + 1
    if (entireImg.shape[0]-INPUT_SIZE) % HI_RE_SIZE != 0:
        imgsPerRow += 1
    numRow = (entireImg.shape[1]-INPUT_SIZE) // HI_RE_SIZE + 1
    if (entireImg.shape[1]-INPUT_SIZE) % HI_RE_SIZE != 0:
        numRow += 1
    rowIdx = idx // imgsPerRow
    colIdx = idx % imgsPerRow
    if rowIdx == numRow - 1:
        x = entireImg.shape[1] - INPUT_SIZE
    else:
        x = rowIdx * HI_RE_SIZE
    if colIdx == imgsPerRow - 1:
        y = entireImg.shape[0] - INPUT_SIZE
    else:
        y = colIdx * HI_RE_SIZE
    return entireImg[y:y+INPUT_SIZE,x:x+INPUT_SIZE,:]



# FETCH a certain number of data from disk
def nextBatchHiRes(batchSize=64):
    random.shuffle(trainSet)
    dataBatch = np.zeros((batchSize, INPUT_SIZE, INPUT_SIZE, NUM_FEATURES+1))
    for i in range(batchSize):
        dataBatch[i,:,:,:] = getImgByIdx(entireImgs[trainSet[i][0]], trainSet[i][1])
    return dataBatch


class ContextEncoder():
    def __init__(self):
        self.img_rows = CROPPED_ROWS
        self.img_cols = CROPPED_COLS
        self.channels = NUM_FEATURES
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=[BinaryAccuracy(), TruePositives(), TrueNegatives(),
                                   FalsePositives(), FalseNegatives()])
        K.get_session().run(tf.local_variables_initializer())

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=5,
                         input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=5, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, kernel_size=5,  padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, kernel_size=5,  padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))


        model.add(Dense(1))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # Mask out the last two channels (congestion east and congestion north)
    def maskHiRes(self, imgs):
        masked_imgs = np.empty(
            (imgs.shape[0], imgs.shape[1], imgs.shape[2], self.channels))
        missing_parts = np.empty((imgs.shape[0]))

        for i, img in enumerate(imgs):
            missing_parts[i] = img[INPUT_SIZE//2-HI_RE_SIZE//2:INPUT_SIZE//2+HI_RE_SIZE//2,INPUT_SIZE//2-HI_RE_SIZE//2:INPUT_SIZE//2+HI_RE_SIZE//2,-1].sum() > 0
            masked_imgs[i] = img[:,:,0:NUM_FEATURES]
        return masked_imgs, missing_parts

    def train(self, iterations, batch_size=128, sample_interval=100):
        testDataCoarse = np.zeros((len(testSet), INPUT_SIZE, INPUT_SIZE, 8))
        for i in range(len(testSet)):
            testDataCoarse[i,:,:,:] = getImgByIdx(entireImgs[testSet[i][0]],testSet[i][1])
        testXCoarse, testYCoarse = self.maskHiRes(testDataCoarse)
        # bestRes = [0, 0, 0, 0]

        for iteration in range(iterations):
            images_train = nextBatchHiRes(batch_size)
            masked_imgs, missing_parts = self.maskHiRes(images_train)
            # Train the discriminator
            weight = {0 : 1, 1: 15}
            d_loss = self.discriminator.train_on_batch(
                masked_imgs, missing_parts, class_weight=weight)

            print("%d [D loss: %f, acc: %f, tp: %d, tn: %d, fp:%d, fn: %d] " %
                  (iteration, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5]))
            if (iteration % 10 == 0):
                logging.info("%d [D loss: %f, acc: %f, tp: %d, tn: %d, fp:%d, fn: %d] " %
                  (iteration, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5]))
            if iteration % sample_interval == 0:
                # self.save_model()
                testLoss = self.test(testXCoarse, testYCoarse)
                logging.info("%d [D loss: %f, acc: %f, tp: %d, tn: %d, fp:%d, fn: %d] Test: [D loss: %f, acc: %f, tp: %d, tn: %d, fp:%d, fn: %d] " %
                (iteration, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5], testLoss[0], testLoss[1], testLoss[2], testLoss[3], testLoss[4], testLoss[5]))

                if testLoss[2] > 2000 and testLoss[4] < testLoss[3]:
                    self.save_model(testLoss[2:])
                    # bestRes = testLoss[2:]
                    logging.info("######## Model saved #########")
                #     continue
                # if testLoss[2] >= bestRes[0] and testLoss[5] <= bestRes[3]:
                #     self.save_model(testLoss[2:])
                #     bestRes = testLoss[2:]
                #     logging.info("######## Model saved #########")


    def save_model(self, metrics):
        def save(model):
            if not os.path.exists("saved_model"):
                os.mkdir("saved_model")
            model_path = "saved_model/%s_%s_%d_%d_%d_%d.json" % (progName[11:-3], lowResName, metrics[0], metrics[1], metrics[2], metrics[3])
            weights_path = "saved_model/%s_%s_%d_%d_%d_%d_weights.hdf5" % (progName[11:-3], lowResName, metrics[0], metrics[1], metrics[2], metrics[3])
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
            model.save('saved_model/%s_%s_%d_%d_%d_%d_keras' % (progName[11:-3], lowResName, metrics[0], metrics[1], metrics[2], metrics[3]))
        save(self.discriminator)
    
    def test(self, testX, testY):
        metrics = self.discriminator.evaluate(testX, testY)
        return metrics


if __name__ == '__main__':
    with open('data/testSetPredCoarseInfo_32.txt', "r") as myfile:
        numPos = 0
        lines = myfile.readlines()
        for line in lines:
            idx, design, status = line.split(' ')
            if status == 'pos\n':
                tileIdxHiRes = getHiResTileByCoarseTile(design, int(idx))
                for tile in tileIdxHiRes:
                    testSet.append((design, tile))
                    img = getImgByIdx(entireImgs[design], tile)
                    if img[INPUT_SIZE//2-HI_RE_SIZE:INPUT_SIZE//2+HI_RE_SIZE,INPUT_SIZE//2-HI_RE_SIZE:INPUT_SIZE//2+HI_RE_SIZE,-1].sum() > 0:
                        numPos += 1
    logging.info('%d test cases, %d pos' % (len(testSet), numPos))
    with open('data/trainSetPredCoarseInfo_32.txt', "r") as myfile:
        numPos = 0
        lines = myfile.readlines()
        for line in lines:
            idx, design, status = line.split(' ')
            if status == 'pos\n':
                tileIdxHiRes = getHiResTileByCoarseTile(design, int(idx))
                for tile in tileIdxHiRes:
                    trainSet.append((design, tile))
    logging.info('%d trainging cases' % (len(trainSet)))

    context_encoder = ContextEncoder()
    time_start = time.perf_counter()
    context_encoder.train(
        iterations=7000, batch_size=128, sample_interval=30)
    print(
        "Time spent on training the model is {0}".format(time.perf_counter()-time_start))
    logging.info("Time spent on training the model is {0}".format(time.perf_counter()-time_start))
    context_encoder.save_model()
