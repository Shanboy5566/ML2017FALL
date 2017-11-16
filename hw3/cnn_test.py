# coding: utf-8
import sys, time
import numpy as np
import csv

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
# from keras.layers import normalization, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import normalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #1080
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #titanx


"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)
"""

"""
# def plot(history, model_name=''):
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title(model_name + ' Model Accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./img/%s-acc.png' % model_name)
#     plt.cla()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title(model_name + ' Model Loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./img/%s-loss.png' % model_name)
#     plt.cla()
"""

def load_data(prefix):
    IN = open(prefix)
    line = IN.readline()
    x_test = []
    
    for line in IN:
        line = line.strip('\n').split(',')
        x_test.append(line[1].split(' '))        
    IN.close()
    
    x_test = np.array(x_test, dtype=float)
    return x_test


def save_result(prefix, Y_test):
    with open(prefix, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('id', 'label'))
        for syaro in range(len(Y_test)):
            writer.writerow((syaro, Y_test[syaro]))


"""
def output_model(model, modelname):
    model.save(modelname)
"""

if __name__ == '__main__':
    argv_1 = sys.argv[1] # location of testing data
    argv_2 = sys.argv[2] # location of predict file
    img_rows, img_cols = 48, 48
    batch_size = 500
    nb_classes = 7
    nb_epoch = 100 # Haley:1000

    # loading testing data
    X_test = load_data(argv_1)
    
    # normalizing testing data, by loading the means_and_stds.csv
    X_test = (X_test * 2 / 255) - 1
    
    # resize X_test
    X_test = np.resize(X_test, (X_test.shape[0], img_rows, img_cols, 1))
    X_test_rev = np.flip(X_test, axis=2)
    
    # loadind model
    model = load_model('./model.h5')
    
    # predict answers
    # Y_test = model.predict_classes(X_test, batch_size=250)
    Y_proba = model.predict(X_test)
    Y_proba_rev = model.predict(X_test_rev)

    ans = []
    for i in range(X_test.shape[0]):
        # ans.append([i])
        Y_proba_list = Y_proba[i].tolist()
        Y_proba_list_rev = Y_proba_rev[i].tolist()
        Y_proba_list_sum = [x+y for x, y in zip(Y_proba_list, Y_proba_list_rev)]

        predict_rev = Y_proba_list_sum.index(max(Y_proba_list_sum))
        ans.append(predict_rev)
    # Y_test = Y_proba.argmax(axis=-1)
    
    # save result
    save_result(argv_2, ans)




