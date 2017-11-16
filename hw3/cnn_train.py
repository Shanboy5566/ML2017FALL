# coding: utf-8
import  csv, json, keras, matplotlib, sys, numpy, pickle, os
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

from keras.utils import np_utils, plot_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #1080
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' #titanx

def plot(history, model_name=''):
    if not os.path.exists('./img'):
        os.makedirs('./img')
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(model_name + ' Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./img/%s-acc.png' % model_name)
    plt.cla()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./img/%s-loss.png' % model_name)
    plt.cla()

def gen_model(nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, generator, is_valid):
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    # earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    
    if(is_valid):
        # history = model.fit_generator(generator, samples_per_epoch=X_train.shape[0]*2, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid), callbacks=[earlyStopping])
        history = model.fit_generator(generator, samples_per_epoch=X_train.shape[0]*2, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
    else:
        history = model.fit_generator(generator, samples_per_epoch=X_train.shape[0]*2, nb_epoch=nb_epoch, validation_data=None, callbacks=[earlyStopping])
    plot(history,'CNN')

    # t = int(time.time())
    print('model release!!')
    if not os.path.exists('./Model'):
        os.makedirs('./Model')
    model.save('./Model/model.h5')
    plot_model(model, to_file='./CNNmodel.png')
    
    return model


def load_data(path):
    with open(path) as fp:
        x_train = []
        y_train = []
        next(fp)
        for line in fp:
            line = line.strip('\n').split(',')
            y_train.append(line[0])
            x_train.append(line[1].split(' '))
    
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    
    return x_train, y_train


def split_data(X_train, Y_train):

    x_train, y_train = [], []
    x_valid, y_valid = [], []

    for i in range(X_train.shape[0]) :
        if(i%valid_ratio == 0):
            x_valid.append(X_train[i])
            y_valid.append(Y_train[i])
        else:
            x_train.append(X_train[i])
            y_train.append(Y_train[i])
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    
    return x_train, y_train, x_valid, y_valid

def ImgGenerator(X_train, Y_train, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode='nearest')
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
    return train_generator

def load_pickle(path):
    file = open(path, 'rb')
    data = pickle.load(file)

    return data

def Normalize(X_train):
    return (X_train / 255) * 2 - 1


if __name__ == '__main__':
    trainData = sys.argv[1] # location of training data
    batch_size = 256
    nb_classes = 7
    nb_epoch = 150 
    is_valid = 1
    valid_ratio = 7

    # loading training data
    X_train, Y_train= load_data(trainData)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)#one hot encoding

    # normalizing training data to -1 ~ 1
    X_train = Normalize(X_train)

    # # resize X_train
    X_train = np.resize(X_train, (X_train.shape[0], 48, 48, 1))

    # split training data to training data and validation data
    if(is_valid):
        X_train, Y_train, X_valid, Y_valid = split_data(X_train, Y_train)
    else:
        X_valid, Y_valid = [], []


    # print('X_train shape = ' , X_train.shape)
    # print('Y_train shape = ' , Y_train.shape)
    # print('X_valid shape = ' , X_valid.shape)
    # print('Y_valid shape = ' , Y_valid.shape)

    # # build an image generator
    generator = ImgGenerator(X_train, Y_train, batch_size)

    # # # train the model
    model = gen_model(nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, generator, is_valid)