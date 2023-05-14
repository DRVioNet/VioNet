from fileinput import filename
import inspect
import numpy as np
import os
import sys

# Size 64 2

hyperImgDir = '/bigdata/ethan/gr-dataset'
vioMapDir = '/bigdata/ethan/wireShortMaps/grSize'
outputDir = '/bigdata/ethan/drv_prediction_dataset_2d_4.0'
enTireImgDir = '/bigdata/ethan/drv_prediction_4.0_entire_hyper_img'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

inputSize = int(sys.argv[1])
outputDir += '/%d' % (inputSize)
if not os.path.exists(enTireImgDir):
    os.mkdir(enTireImgDir)
if not os.path.exists(outputDir):
    os.mkdir(outputDir)
    
count = 0
for fileName in os.listdir(hyperImgDir):
    fileCount = 0
    pos, neg = 0, 0
    designName = fileName.split('_test')[0][-1] + 't' + fileName.split('_test')[1].split('.npy')[0]
    if designName == '9t4' or designName == '9t5' or designName[0] == '8':
        continue
    print(designName)
    # hyperImgOutputDir = outputDir + '/hyperImages/' + designName
    # vioMapOutputDir = outputDir + '/vioMaps/' + designName
    designDir = outputDir + '/' + designName
    if not os.path.exists(designDir):
        os.mkdir(designDir)
    deleted = 0
    for f in os.listdir(designDir):
        os.remove(os.path.join(designDir, f))
        deleted += 1
    print('%d files deleted' % (deleted))

    hyperImage = np.load(hyperImgDir + '/' + fileName)[:,:,:23]
    print(hyperImage.shape)
    vioMap = np.load(vioMapDir + '/wireShortMap_grSize_%s.npy' % designName).transpose(1,2,0)
    print(vioMap.shape)
    vioMap2D = np.zeros((vioMap.shape[0], vioMap.shape[1], 1))
    vioMap2D[:,:,0] = np.sum(vioMap, axis = 2)
    featureMaps = np.zeros((hyperImage.shape[0], hyperImage.shape[1], 6))
    featureMaps[:,:,:5] = hyperImage[:,:,:5]
    featureMaps[:,:,5] = np.sum(hyperImage[:,:,5:23], axis=2) / 18
    entireImage = np.concatenate((featureMaps, vioMap2D), axis=2)
    np.save(os.path.join('/bigdata/ethan/drv_prediction_entire_hyper_img_no_padding', '%s.npy' % designName), entireImage)
    entireImage = np.pad(entireImage, ((31 , 31), (31, 31), (0, 0)), 'edge')
    np.save(os.path.join(enTireImgDir, '%s.npy' % designName), entireImage)

    print('shape after padding: %d, %d, %d' % (entireImage.shape[0], entireImage.shape[1], entireImage.shape[2]))
    for i in range(0, entireImage.shape[0]-inputSize+1, 2):
        for j in range(0, entireImage.shape[1]-inputSize+1, 2):
            croppedImg = entireImage[i:i+inputSize,j:j+inputSize,:]
            if np.sum(croppedImg[31:33,31:33,6]) > 0:
                pos += 1
            else:
                neg += 1
            np.save(designDir + '/%d.npy' % (count), croppedImg)
            count += 1
            fileCount += 1
            if (count % 50000 == 0):
                print('%d images done, pos: %d, neg: %d' % (count, pos, neg))

    if (entireImage.shape[0]-inputSize) % 2 != 0:
        print('shape[0] is odd')
        for j in range(0, entireImage.shape[1]-inputSize+1, 2):
            croppedImg = entireImage[entireImage.shape[0]-inputSize:entireImage.shape[0],j:j+inputSize,:]
            if np.sum(croppedImg[31:33,31:33,6]) > 0:
                pos += 1
            else:
                neg += 1
            np.save(designDir + '/%d.npy' % (count), croppedImg)
            count += 1
            fileCount += 1
            if (count % 50000 == 0):
                print('%d images done, pos: %d, neg: %d' % (count, pos, neg))
    if (entireImage.shape[1]-inputSize) % 2 != 0:
        print('shape[1] is odd')
        for i in range(0, entireImage.shape[0]-inputSize+1, 2):
            croppedImg = entireImage[i:i+inputSize,entireImage.shape[1]-inputSize:entireImage.shape[1],:]
            if np.sum(croppedImg[31:33,31:33,6]) > 0:
                pos += 1
            else:
                neg += 1
            np.save(designDir + '/%d.npy' % (count), croppedImg)
            count += 1
            fileCount += 1
            if (count % 50000 == 0):
                print('%d images done, pos: %d, neg: %d' % (count, pos, neg))

    print('%d imgs for %s, pos %d, neg %d' % (fileCount, filename, pos,neg))