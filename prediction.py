from __future__ import print_function, division
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
from keras.models import model_from_json, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.metrics import BinaryAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

import numpy as np
import math
import logging
from datetime import date, datetime
import random

MODEl_DIR       = "./saved_model"
entireImgDir    = '/bigdata/ethan/drv_prediction_entire_hyper_img_no_padding'
modelName = sys.argv[1]

# Logging
progName = sys.argv[0].split('/')[-1]
today = str(date.today())
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
logFileName = progName[0:-3] + "-" + today + '-' + modelName + '-' + current_time + ".log"
if not os.path.exists('./log'):
    os.mkdir('./log')
logging.basicConfig(filename='./log/' + logFileName, level=logging.DEBUG)
logging.info('Start')
logging.info('')

INPUT_SIZE = 64  # size of input box to the model
# This is pixel height, make sure height and width are the same
CROPPED_ROWS = INPUT_SIZE
CROPPED_COLS = INPUT_SIZE  # This is pixel width
COARSE_SIZE = 64
HI_RE_SIZE = 4
PAD_SIZE = int((COARSE_SIZE - HI_RE_SIZE) / 2)
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
numFileCoarse = 1873 

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


class MyModel:
    def __init__(self):
        model_path = MODEl_DIR + "/%s.json" % (modelName)
        weights_path = MODEl_DIR + "/%s_weights.hdf5" % (modelName)
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_file = open(options['file_arch'], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(options['file_weight'])
        self.model.compile(loss='mse',
                                    optimizer=Adam(0.002),
                                    metrics=[BinaryAccuracy(), TruePositives(), TrueNegatives(),
                                    FalsePositives(), FalseNegatives()])


    def doTest(self, inputX, targetY):
        metrics = self.model.evaluate(inputX, targetY)
        return metrics

def maskHiRes(imgs):
    masked_imgs = np.empty(
        (imgs.shape[0], imgs.shape[1], imgs.shape[2], NUM_FEATURES))
    missing_parts = np.empty((imgs.shape[0]))

    for i, img in enumerate(imgs):
        missing_parts[i] = img[INPUT_SIZE//2-HI_RE_SIZE//2:INPUT_SIZE//2+HI_RE_SIZE//2,INPUT_SIZE//2-HI_RE_SIZE//2:INPUT_SIZE//2+HI_RE_SIZE//2,-1].sum() > 0
        masked_imgs[i] = img[:,:,0:NUM_FEATURES]
    return masked_imgs, missing_parts

    
if __name__ == '__main__':
    myModel = MyModel()

    testCases = {}
    trainingCases = {}
    numTrainingTN = {
        '9t1' : 0,
        '9t2' : 20*256 + 2*256-6,
        '9t3' : 1*256 + 0,
        '9t6' : 94*256 + 8*256-23,
        '9t7' : 104*256 + 3*256-5,
        '9t8' : 80*256 + 7*256-7,
        '9t9' : 98*256 + 3*256-3,
        '9t10' : 98*256 + 1*256-1
    }
    numTrainingFN = {
        '9t1' : 0,
        '9t2' : 6,
        '9t3' : 0,
        '9t6' : 23,
        '9t7' : 5,
        '9t8' : 7,
        '9t9' : 3,
        '9t10' : 1
    }
    numTestTN = {
        '9t1' : 0,
        '9t2' : 3*256 + 1*256-1,
        '9t3' : 0,
        '9t6' : 21*256 + 1*256-1,
        '9t7' : 37*256 + 1*256-1,
        '9t8' : 21*256 + 1*256-1,
        '9t9' : 25*256,
        '9t10' : 26*256 + 1*256-3
    }
    numTestFN = {
        '9t1' : 0,
        '9t2' : 1,
        '9t3' : 0,
        '9t6' : 1,
        '9t7' : 1,
        '9t8' : 1,
        '9t9' : 0,
        '9t10' : 3
    }
    for design in designs:
        testCases[design] = []
        trainingCases[design] = []

    with open('data/testSetPredCoarseInfo_64.txt', "r") as myfile:
        lines = myfile.readlines()
        for line in lines:
            idx, design, status = line.split(' ')
            if status == 'pos\n':
                tileIdxHiRes = getHiResTileByCoarseTile(design, int(idx))
                testCases[design] += tileIdxHiRes
    with open('data/trainSetPredCoarseInfo_64.txt', "r") as myfile:
        lines = myfile.readlines()
        for line in lines:
            idx, design, status = line.split(' ')
            if status == 'pos\n':
                tileIdxHiRes = getHiResTileByCoarseTile(design, int(idx))
                trainingCases[design] += tileIdxHiRes
    
    for design in designs:
        print('##############  Start %s ##################' % design)
        print('Trainging set')
        logging.info('##############  Start %s ##################' % design)
        logging.info('Trainging set')
        numPos = 0

        inputTraining = np.zeros((len(trainingCases[design]), INPUT_SIZE, INPUT_SIZE, 8))
        for i in range(len(trainingCases[design])):
            inputTraining[i,:,:,:] = getImgByIdx(entireImgs[design],trainingCases[design][i])
            img = getImgByIdx(entireImgs[design],trainingCases[design][i])
            if img[30:34,30:34,-1].sum() > 0:
                numPos += 1
        print('%d pos' % numPos)

        x, y = maskHiRes(inputTraining)
        metrics = myModel.doTest(x, y)

        print('acc: %f, tp: %d, tn: %d, fp:%d, fn: %d' % (metrics[1], metrics[2], metrics[3]+numTrainingTN[design], metrics[4], metrics[5]+numTrainingFN[design]))
        logging.info('acc: %f, tp: %d, tn: %d, fp:%d, fn: %d' % (metrics[1], metrics[2], metrics[3]+numTrainingTN[design], metrics[4], metrics[5]+numTrainingFN[design]))

        print('Test set')
        logging.info('Test set')

        numPos = 0
        inputTest = np.zeros((len(testCases[design]), INPUT_SIZE, INPUT_SIZE, 8))
        if len(testCases[design]) == 0:
            continue
        for i in range(len(testCases[design])):
            inputTest[i,:,:,:] = getImgByIdx(entireImgs[design],testCases[design][i])
            img = getImgByIdx(entireImgs[design],testCases[design][i])
            if img[30:34,30:34,-1].sum() > 0:
                numPos += 1
        print('%d pos' % numPos)

        testX, testY = maskHiRes(inputTest)
        metrics = myModel.doTest(testX, testY)
        print('acc: %f, tp: %d, tn: %d, fp:%d, fn: %d' % (metrics[1], metrics[2], metrics[3]+numTestTN[design], metrics[4], metrics[5]+numTestFN[design]))
        logging.info('acc: %f, tp: %d, tn: %d, fp:%d, fn: %d' % (metrics[1], metrics[2], metrics[3]+numTestTN[design], metrics[4], metrics[5]+numTestFN[design]))




