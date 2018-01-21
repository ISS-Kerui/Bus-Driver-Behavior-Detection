# -*- coding: utf-8 -*-

"""
*********************************************************
author: ISS-Kerui
date:2018-01-20
VGG_16 is a function to extracted features from raw pictures using VGG-16 model.
Bi_LSTM is a function to using Bi-LSTM to combine 50 pictures' feature(we cut each video file into 50 frames) maps and get the final result.
vgg_image_feature is a function to reszie and standardized pictures.
*********************************************************
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Activation,TimeDistributed,Bidirectional,Input
from keras.layers.convolutional import AveragePooling1D
from keras.layers.recurrent import GRU,LSTM
import log_IBRD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import theano
from keras.utils import np_utils
import cv2,numpy as np        
import cPickle
from multiprocessing import Pool
import os
import time
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
import sys
reload(sys)   
sys.setdefaultencoding('utf8') 
def VGG_16(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten(name='flat'))
    model.add(Dense(4096, activation='relu',name = 'dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

import theano
from keras.utils import np_utils

model_vgg16 = VGG_16('../h5/vgg16_weights.h5')
layer_dict = dict([(layer.name, layer) for layer in model_vgg16.layers])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy')    
layer = layer_dict['flat']
transformer = theano.function([model_vgg16.input], layer.output)


def vgg_image_feature(img_path):
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)#读入图片BGR通道
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1)) # [103.939, 116.779, 123.68]BGR
    im = np.expand_dims(im, axis=0)
    return transformer(im).reshape(-1)

def Bi_LSTM(weights_path,test_data,test_labels,test_vname):
    model_IBRD = Sequential()

    model_IBRD.add(TimeDistributed(Dense(256),input_shape=(50,25088)))

    model_IBRD.add(Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave'))

    #model_IBRD.add(TimeDistributed(Dense(8,activation='softmax')))
    model_IBRD.add(AveragePooling1D(pool_length=50))

    model_IBRD.add(Flatten())

    model_IBRD.add(Dropout(0.5))

    model_IBRD.add(Dense(8))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False) 
    #adam = Adam(lr=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
    model_IBRD.add(Activation('softmax'))
    model_IBRD.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    model_IBRD.load_weights(weights_path)

    pred_y = model_IBRD.predict(test_data)
    count = 0
    normal_pdict_value = []
    phone_pdict_value = []
    smoke_pdict_value = []
    seatfree_pdict_value = []
    pred_classes = pred_y
    for resultX in range(pred_y.shape[0]):
        m = max(pred_y[resultX])
        for resultY in range(pred_y.shape[1]):
            if(m != pred_y[resultX][resultY]):
                pred_classes[resultX][resultY] = 0
            else:
                pred_classes[resultX][resultY] = 1
    for label in test_labels:
        if (label == np.array([1,0,0,0,0,0,0,0])).all():
            if (label == pred_classes[count]).all() == False:
                normal_pdict_value.append(pred_y[count])
        elif (label == np.array([0,1,0,0,0,0,0,0])).all():
            if (label == pred_classes[count]).all() == False:
                phone_pdict_value.append(pred_y[count])
        elif (label == np.array([0,0,1,0,0,0,0,0])).all():
            if (label == pred_classes[count]).all() == False:
                smoke_pdict_value.append(pred_y[count])
        elif (label == np.array([0,0,0,0,0,0,0,1])).all():
            if (label == pred_classes[count]).all() == False:
                seatfree_pdict_value.append(pred_y[count])
        count += 1

    normal_pdict_value = np.array(normal_pdict_value)
    phone_pdict_value = np.array(phone_pdict_value)
    smoke_pdict_value = np.array(smoke_pdict_value)
    seatfree_pdict_value = np.array(seatfree_pdict_value)
    np.save("predict_value/normal.npy",normal_pdict_value)
    np.save("predict_value/phone.npy",phone_pdict_value)
    np.save("predict_value/smoke.npy",smoke_pdict_value)
    np.save("predict_value/seatfree.npy",seatfree_pdict_value)

   

    logger.info('the labels...')
    logger.info(test_labels)
    logger.info('save the pred_y ...')
    logger.info(pred_y)
    



    
    
    

    logger.info('save the pred_y(dealed) ...')
    logger.info(pred_y)
    ratio0 = 0.0
    ratio0 = pred_y*test_labels
    ratio0 = ratio0.sum()/ratio0.shape[0]


    logger.info(u'accuracy ： %f' %ratio0)
  


from multiprocessing import Pool
import os


import theano
from keras.utils import np_utils

model_vgg16 = VGG_16('../h5/vgg16_weights.h5')
layer_dict = dict([(layer.name, layer) for layer in model_vgg16.layers])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_vgg16.compile(optimizer=sgd, loss='categorical_crossentropy')    
layer = layer_dict['flat']
transformer = theano.function([model_vgg16.input], layer.output)
if __name__ == "__main__":
    logger = log_IBRD.Logger(logsite='experimentTest').getlog()
    test_data = []
    test_labels = []
    test_vname = []
    test_file = '../dataset/img2/Test-Image'
    time_whole = time.time()
    logger.info('load and analyze the features of test data ... ')
    for label in os.listdir(test_file):
        current_videos = os.path.join(test_file,label)        
        for f in os.listdir(current_videos):
            
            current_video = os.path.join(current_videos,f)
            test_vname.append(current_video)
            time_current_test = time.time()
            logger.info('    video of '+current_video+' ...')
           
            video=[]
            for img_file in os.listdir(current_video):
                img = os.path.join(current_video,img_file)
                img = img.replace('\\', '/')
                video.append(img)
            
            video = sorted(video,key=lambda a: int(a.split('_')[-1].split('.')[0]))
#            for i in range(len(video)):
#                video[i] = (convout1_f, video[i])
#            print video
            p_test = Pool()
            #from multiprocessing import cpu_count
            #p_test = Pool(int(0.75*cpu_count()))
            temp = p_test.map(vgg_image_feature, video)
            p_test.close()
            p_test.join()
            current_time = time.time()
            logger.info('        done in %d min %d s .'%((current_time-time_current_test)/60, (current_time-time_current_test)%60))
            #print '        done in %d min %d s .'%((current_time-time_current_test)/60, (current_time-time_current_test)%60)
            
            video_features = np.asarray(temp)
            test_data.append(video_features)
            if label == 'Normal':
                test_labels.append(0)
            elif label == 'Phone':
                test_labels.append(1)
            elif label == 'Smoke':
                test_labels.append(2)
            elif label == 'Eat':
                test_labels.append(3)
            elif label == 'Screencovered':
                test_labels.append(4)
            elif label == 'Screenblurred':
                test_labels.append(5)
            elif label == 'Blackscreen':
                test_labels.append(6)
            else:
            #elif label == 'Seatfree':
                test_labels.append(7)
                
    test_data = np.asarray(test_data)
    test_labels = np_utils.to_categorical(test_labels, 8)

    np.save("../npy/test_data.npy", test_data)
    np.save("../npy/test_labels.npy", test_labels)
    current_time = time.time()
  
    logger.info('analyzing test data has taken %d h %d min %d s .'%((current_time-time_whole)/3600, ((current_time-time_whole)%3600)/60, (current_time-time_whole)%60))
 
    Bi_LSTM('../h5/IBRD_weights.h5',test_data,test_labels,test_vname)

