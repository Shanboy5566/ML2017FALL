"""
Normal import
"""
import numpy as np
from random import randint
import os
import sys
import tensorflow as tf
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
# import pickle
from pyfasttext import FastText
# from gensim.models.keyedvectors import KeyedVectors

"""
GPU Area
"""
# import seq2seq
# from seq2seq.models import SimpleSeq2Seq
# from seq2seq.models import AttentionSeq2Seq

import keras.backend as K
from keras.objectives import cosine_proximity
from keras import regularizers
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU, Dense, RepeatVector, Bidirectional
from keras.models import Model, Sequential
from keras.layers import Flatten, Input, Add, Dot, Concatenate, Dense, Dropout, merge, Reshape, Lambda, Activation, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Titan
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 1080

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""
Global var
"""
input_length = 246
input_dim = 39

output_length = 13
output_dim = 300

samples = 45036
hidden_dim = 24

"""
Self define
"""


def paddingZero(Labels):
    tmp = []
    npz = np.zeros((13, 300))
    for e in Labels:
        x = np.copy(npz)
        x[:len(e)] = e
        tmp.append(x)
    Arr = np.array(tmp)
    return Arr


def saveHistory(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.save('./result/history.png')


def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)


"""
Model compile
"""


# def seq2seq(hidden_dim, output_length, output_dim):
#     # model = Sequential()
#     # model.add(Bidirectional(LSTM(hidden_dim, return_sequences=False),
#     #                         input_shape=(13, 300)))
#     # model.add(Dense(hidden_dim, activation="relu"))
#     # model.add(RepeatVector(output_length))
#     # model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
#     # model.add(TimeDistributed(Dense(output_dim=output_dim, activation="linear")))
#     # model.compile(loss='mse', optimizer='adam')

#     # model.summary()
#     # Define an input sequence and process it.
#     encoder_inputs = Input(shape=(None, 13, 100))
#     encoder = LSTM(256, return_state=True)
#     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#     # We discard `encoder_outputs` and only keep the states.
#     encoder_states = [state_h, state_c]

#     # Set up the decoder, using `encoder_states` as initial state.
#     decoder_inputs = Input(shape=(None, 13, 100))
#     # We set up our decoder to return full output sequences,
#     # and to return internal states as well. We don't use the
#     # return states in the training model, but we will use them in inference.
#     decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                          initial_state=encoder_states)
#     decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#     decoder_outputs = decoder_dense(decoder_outputs)

#     # Define the model that will turn
#     # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#     return model


# def gen_model(n_mfccs, n_texts, latent_dim=20):
#     mfcc_input = Input(shape=[246, 39])
#     text_input = Input(shape=[13])

#     hidden = LSTM(64)(mfcc_input)
#     hidden = Dense(128)(hidden)
#     mfcc_output = Dense(256)(hidden)
#     # encoder_flat = Flatten()(encoder)
#     # hidden = Dense(64, activation="sigmoid")(encoder_flat)
#     # hidden = Dropout(0.3)(hidden)
#     # hidden = Dense(32, activation="sigmoid")(hidden)
#     # hidden = Dropout(0.3)(hidden)
#     # mfcc_output = Dense(13, activation="sigmoid")(hidden)
#     # mfcc_output = Flatten()(hidden)

#     text_input_emb = Embedding(n_texts, latent_dim,
#                                embeddings_initializer='uniform')(text_input)
#     hidden = LSTM(64)(text_input_emb)
#     hidden = Dense(128)(hidden)
#     text_output = Dense(256)(hidden)
#     # text_output = Dense(13, activation="sigmoid")(encoder)
#     # text_output = Flatten()(hidden)

#     r_hat = Dot(axes=1)([mfcc_output, text_output])

#     output = Dense(1, activation='sigmoid')(r_hat)

#     model = Model([mfcc_input, text_input], output)
#     model.compile(loss='binary_crossentropy', optimizer='adamax')
#     model.summary()

#     return model


def gen_model_pretrain_LSTM():
    mfcc_input = Input(shape=[246, 39])
    text_input = Input(shape=[13, 300])

    hidden = LSTM(256, return_sequences=True, dropout=0.3,
                  recurrent_dropout=0.3)(mfcc_input)
    # hidden = LSTM(256, return_sequences=True, dropout=0.2,
    #               recurrent_dropout=0.2)(hidden)
    hidden = LSTM(256, dropout=0.3, recurrent_dropout=0.3)(hidden)
    mfcc_output = Dense(512)(hidden)

    hidden = LSTM(256, return_sequences=True, dropout=0.3,
                  recurrent_dropout=0.3)(text_input)
    # hidden = LSTM(256, return_sequences=True, dropout=0.2,
    #               recurrent_dropout=0.2)(hidden)
    hidden = LSTM(256, dropout=0.3, recurrent_dropout=0.3)(hidden)
    text_output = Dense(512)(hidden)

    r_hat = Dot(axes=1)([mfcc_output, text_output])  # merge layer
    # r_hat = concatenate([mfcc_output, text_output])

    output = Dense(1, activation='sigmoid')(r_hat)

    model = Model([mfcc_input, text_input], output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()

    return model


def gen_model_pretrain_GRU():
    mfcc_input = Input(shape=[246, 39])
    text_input = Input(shape=[13, 300])

    hidden = GRU(256, return_sequences=True, dropout=0.3,
                 recurrent_dropout=0.3)(mfcc_input)
    # hidden = LSTM(256, return_sequences=True, dropout=0.2,
    #               recurrent_dropout=0.2)(hidden)
    hidden = GRU(256, dropout=0.3, recurrent_dropout=0.3)(hidden)
    mfcc_output = Dense(512)(hidden)

    hidden = GRU(256, return_sequences=True, dropout=0.3,
                 recurrent_dropout=0.3)(text_input)
    # hidden = LSTM(256, return_sequences=True, dropout=0.2,
    #               recurrent_dropout=0.2)(hidden)
    hidden = GRU(256, dropout=0.3, recurrent_dropout=0.3)(hidden)
    text_output = Dense(512)(hidden)

    r_hat = Dot(axes=1)([mfcc_output, text_output])  # merge layer
    # r_hat = concatenate([mfcc_output, text_output])

    output = Dense(1, activation='sigmoid')(r_hat)

    model = Model([mfcc_input, text_input], output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()

    return model


"""
Load Data
X_train : MFCC
X_train shape = (45036, 13, 300)
"""

# X = np.load("/media/Data/Shanboy/final/data/train.data")
# npz = np.zeros((input_length, input_dim))

# X_train = []
# for e in X:
#     x = np.copy(npz)
#     x[:len(e)] = e
#     X_train.append(x)

# X_train = np.array(X_train)

# print('X_train shape = ', X_train.shape)

#####juice######

# X = np.load('/media/Data/Shanboy/final/data/MFCC_cnn_train/CNN_trainOutput_0.npy')
# for i in range(1, 23):
#     tmp = np.load(
#         '/media/Data/Shanboy/final/data/MFCC_cnn_train/CNN_trainOutput_' + str(i) + '.npy')
#     X = np.concatenate((X, tmp), axis=0)

# np.save('/media/Data/Shanboy/final/data/MFCC_cnn_train/X_train.npy', X)

# X_train = np.load('/media/Data/Shanboy/final/data/MFCC_cnn_train/X_train.npy')
# print('X_train shape = ', X_train.shape)
# print('X_train[0] = ', X_train[0])


"""
Load Data(A)
Y_train : Caption
For pretrain model
"""
# model = FastText('./wiki.zh/wiki.zh.bin')

# Y_train = []
# with (open("/media/Data/Shanboy/final/data/train.caption", "r")) as openfile:
#     content = openfile.readlines()
# content = [x.strip() for x in content]
# Y_train = np.array(content)

# Labels = []
# for words in Y_train:
#     words = words.split(' ')
#     Label = []
#     for word in words:
#         # try:
#         v = model[word]
#         Label.append(v)
#         # except KeyError:
#         #     continue
#     Labels.append(Label)

# # print(np.array(Labels).shape)

# Y_train = paddingZero(Labels)

# print('Y_train shape = ', Y_train.shape)
# print('Y_train[0][2] = ', Y_train[0][2])
# print('Y_train[0][3] = ', Y_train[0][3])

#####juice######

# Y_train = np.load('/media/Data/Shanboy/final/data/g_truth/y_data.npy')
# Y_train.resize(45036, 13, 300)
# print('Y_train shape = ', Y_train.shape)  # (45036, 13, 300)
# print('Y_train[0] = ', Y_train[0])
# Y_train_target = np.roll(Y_train, -1)

"""
Load Data(B)
Y_train : Caption
For no pretrain model
"""
# Y_train = []
# with (open("/media/Data/Shanboy/final/data/train.caption", "r")) as openfile:
#     content = openfile.readlines()
# content = [x.strip() for x in content]

"""
Load Data(B)
Tokenizer
"""

# from keras.preprocessing.text import Tokenizer
# nb_words = 10000
# tokenizer = Tokenizer(num_words=nb_words)
# tokenizer.fit_on_texts(content)
# Text = tokenizer.texts_to_sequences(content)
# print(np.array(Text).shape)

# tmp = []
# npz = np.zeros((13))
# for e in Text:
#     x = np.copy(npz)
#     x[:len(e)] = e
#     tmp.append(x)

# Y_train = np.array(tmp)
# print(Y_train.shape)

"""
X_train.shape = (45036, 246, 39)
Y_train.shape = (45036, 13, 300) with pre-train model
Y_train.shape = (45036, 13) with no pre-train model
"""
# n_mfccs = X_train.shape[0]
# n_texts = X_train.shape[0]

"""
Make more data
"""
# np.random.seed(66)
# n_data = n_mfccs * 4

# xTrainfake1 = X_train
# xTrainfake2 = X_train
# xTrainfake3 = X_train
# MFCC_data = np.concatenate(
#     (X_train, xTrainfake1, xTrainfake2, xTrainfake3), axis=0)

# yTrainfake1 = np.roll(Y_train, 2, axis=0)
# yTrainfake2 = np.roll(Y_train, 4, axis=0)
# yTrainfake3 = np.roll(Y_train, 6, axis=0)
# Text_data = np.concatenate(
#     (Y_train, yTrainfake1, yTrainfake2, yTrainfake3), axis=0)

# Y_data = np.ones(n_data)
# Y_data[n_mfccs:] = 0

###juice###

# yTrainfake1 = []
# yTrainfake2 = []
# yTrainfake3 = []
# for word in Y_train:
#     yTrainfake1.append(np.random.permutation(word))
#     yTrainfake2.append(np.random.permutation(word))
#     yTrainfake3.append(np.random.permutation(word))
# yTrain_fake1 = np.array(yTrainfake1)
# yTrain_fake2 = np.array(yTrainfake2)
# yTrain_fake3 = np.array(yTrainfake3)

# yTrain_fake1 = paddingZero(yTrainfake1)
# yTrain_fake2 = paddingZero(yTrainfake2)
# yTrain_fake3 = paddingZero(yTrainfake3)

# text_input_1 = []
# text_input_2 = []
# text_input_3 = []
# text_input_4 = []
# ground_truth = []

# for i in range(0, n_mfccs):
#     gt = [1, 0, 0, 0]
#     texts = []
#     texts.append(Y_train[i])
#     texts.append(yTrain_fake1[i])
#     texts.append(yTrain_fake2[i])
#     texts.append(yTrain_fake3[i])

#     gt = np.array(gt)
#     texts = np.array(texts)

#     randomize = np.arange(len(gt))
#     np.random.shuffle(randomize)
#     gt = gt[randomize]
#     texts = texts[randomize]

#     text_input_1.append(texts[0])
#     text_input_2.append(texts[1])
#     text_input_3.append(texts[2])
#     text_input_4.append(texts[3])

#     ground_truth.append(gt)


# MFCC_data = X_train
# text_input_1 = np.array(text_input_1)
# text_input_2 = np.array(text_input_2)
# text_input_3 = np.array(text_input_3)
# text_input_4 = np.array(text_input_4)
# ground_truth = np.array(ground_truth)


# # Text_data = np.concatenate(
# #     (Y_train, yTrain_fake1, yTrain_fake2, yTrain_fake3), axis=0)
# # Y_data = np.ones(n_data)
# # Y_data[n_mfccs:] = 0

"""
Shuffle the data
"""

# indice = np.random.permutation(n_data)
# MFCC_data = MFCC_data[indice]
# Text_data = Text_data[indice]
# Y_data = Y_data[indice]
# print('Y_data shape = ', Y_data.shape)
# print('Y_data[0] = ', Y_data[0])
# print('Y_data[1] = ', Y_data[1])
# print('Y_data[2] = ', Y_data[2])

## For seq2seq ###
# indice_s2s = np.random.permutation(n_mfccs)
# X_train = X_train[indice_s2s]
# Y_train = Y_train[indice_s2s]
# Y_train_target = Y_train_target[indice_s2s]

"""
Split to train & validation

(A) With Pretrain (True : Fake = 1 : 1)
MFCC_train shape =  (81065, 246, 39)
Text_train shape =  (81065, 13, 300)
Y_train shape =  (81065,)
MFCC_val shape =  (9007, 246, 39)
Text_val shape =  (9007, 13, 300)
Y_val shape =  (9007,)

(B) No Pretrain

"""

# split_ratio = 0.10
# num_validation_sample = int(split_ratio * n_data)

# MFCC_train = MFCC_data[num_validation_sample:]
# Text_train = Text_data[num_validation_sample:]
# Label_train = Y_data[num_validation_sample:]

# MFCC_val = MFCC_data[:num_validation_sample]
# Text_val = Text_data[:num_validation_sample]
# Label_val = Y_data[:num_validation_sample]


# print('MFCC_train shape = ', MFCC_train.shape)
# print('Text_train shape = ', Text_train.shape)
# print('Y_train shape = ', Label_train.shape)

# print('MFCC_val shape = ', MFCC_val.shape)
# print('Text_val shape = ', Text_val.shape)
# print('Y_val shape = ', Label_val.shape)


### For seq2seq ###
# num_validation_sample = int(split_ratio * n_mfccs)
# en_Train = X_train[num_validation_sample:]
# en_Val = X_train[:num_validation_sample]
# de_Train = Y_train[num_validation_sample:]
# de_Val = Y_train[:num_validation_sample]
# tg_Train = Y_train_target[num_validation_sample:]
# tg_Val = Y_train_target[:num_validation_sample]

# print('xTrain shape = ', xTrain.shape)
# print('yTrain shape = ', yTrain.shape)

# print('xVal shape = ', xVal.shape)
# print('yVal shape = ', yVal.shape)

####juice####

# print('MFCC_data shape = ', MFCC_data.shape)
# print('texts_input_1 shape = ', text_input_1.shape)
# print('texts_input_2 shape = ', text_input_2.shape)
# print('texts_input_3 shape = ', text_input_3.shape)
# print('texts_input_4 shape = ', text_input_4.shape)
# print('ground_truth shape = ', ground_truth.shape)

# split_ratio = 0.10
# num_validation_sample = int(split_ratio * n_mfccs)

# MFCC_train = MFCC_data[num_validation_sample:]
# texts_input_1_train = text_input_1[num_validation_sample:]
# texts_input_2_train = text_input_2[num_validation_sample:]
# texts_input_3_train = text_input_3[num_validation_sample:]
# texts_input_4_train = text_input_4[num_validation_sample:]
# ground_truth_train = ground_truth[num_validation_sample:]

# MFCC_val = MFCC_data[:num_validation_sample]
# texts_input_1_val = text_input_1[:num_validation_sample]
# texts_input_2_val = text_input_2[:num_validation_sample]
# texts_input_3_val = text_input_3[:num_validation_sample]
# texts_input_4_val = text_input_4[:num_validation_sample]
# ground_truth_val = ground_truth[:num_validation_sample]

# print('MFCC_train shape = ', MFCC_train.shape)
# print('texts_input_1_train shape = ', texts_input_1_train.shape)
# print('ground_truth_train shape = ', ground_truth_train.shape)

# print('MFCC_val shape = ', MFCC_val.shape)
# print('texts_input_1_val shape = ', texts_input_1_val.shape)
# print('ground_truth_val shape = ', ground_truth_val.shape)

"""
Save Train & Val npy
Now Fake : True = 3 : 1
MFCC_train shape =  (162130, 246, 39)
Text_train shape =  (162130, 13, 300)
Y_train shape =  (162130,)
MFCC_val shape =  (18014, 246, 39)
Text_val shape =  (18014, 13, 300)
Y_val shape =  (18014,)
"""
# if not os.path.exists('/media/Data/Shanboy/final/data/New_work'):
#     os.makedirs('/media/Data/Shanboy/final/data/New_work')

# np.save('/media/Data/Shanboy/final/data/New_work/MFCC_train.npy', MFCC_train)
# np.save('/media/Data/Shanboy/final/data/New_work/Text_train.npy', Text_train)
# np.save('/media/Data/Shanboy/final/data/New_work/Label_train.npy', Label_train)

# np.save('/media/Data/Shanboy/final/data/New_work/MFCC_val.npy', MFCC_val)
# np.save('/media/Data/Shanboy/final/data/New_work/Text_val.npy', Text_val)
# np.save('/media/Data/Shanboy/final/data/New_work/Label_val.npy', Label_val)

"""
Load Train & Val npy
"""

MFCC_train = np.load('/media/Data/Shanboy/final/data/New_work/MFCC_train.npy')
Text_train = np.load('/media/Data/Shanboy/final/data/New_work/Text_train.npy')
Label_train = np.load(
    '/media/Data/Shanboy/final/data/New_work/Label_train.npy')

MFCC_val = np.load('/media/Data/Shanboy/final/data/New_work/MFCC_val.npy')
Text_val = np.load('/media/Data/Shanboy/final/data/New_work/Text_val.npy')
Label_val = np.load('/media/Data/Shanboy/final/data/New_work/Label_val.npy')


"""
Train model
"""

model_path = sys.argv[1]

# # model = AttentionSeq2Seq(input_dim=input_dim, input_length=input_length,
# #                          hidden_dim=10, output_length=output_length, output_dim=output_dim, depth=4)
# # model.compile(loss='mse', optimizer='rmsprop')

# model = seq2seq(hidden_dim=256, output_length=output_length,
#                 output_dim=output_dim)
# model = gen_model(n_mfccs, n_texts, latent_dim=300)
# model = gen_model_pretrain_LSTM()
model = gen_model_pretrain_GRU()

earlystopping = EarlyStopping(
    monitor='val_loss', patience=3, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath=model_path,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')

# history = model.fit(xTrain, yTrain, epochs=100, validation_data=(xVal, yVal),
#                     callbacks=[earlystopping, checkpoint], batch_size=256)

history = model.fit([MFCC_train, Text_train], Label_train, epochs=100, validation_data=([MFCC_val, Text_val], Label_val),
                    callbacks=[earlystopping, checkpoint], batch_size=256)

####juice#####
# model = gen_model_pretrain_new(n_mfccs, n_texts)

# earlystopping = EarlyStopping(
#     monitor='val_loss', patience=1, verbose=1, mode='auto')
# checkpoint = ModelCheckpoint(filepath=model_path,
#                              verbose=1,
#                              save_best_only=True,
#                              monitor='val_loss',
#                              mode='min')
# history = model.fit([MFCC_train, texts_input_1_train, texts_input_2_train, texts_input_3_train, texts_input_4_train], ground_truth_train, epochs=100, validation_data=([MFCC_val, texts_input_1_val, texts_input_2_val, texts_input_3_val, texts_input_4_val], ground_truth_val),
#                     callbacks=[earlystopping, checkpoint], batch_size=256)

# saveHistory(history)

# if not os.path.exists('./model'):
#     os.makedirs('./model')
# model.save("./model/model_cross.h5")
