"""
Normal import
"""
import numpy as np
from random import randint
import os
import sys
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from pyfasttext import FastText

"""
GPU Area
"""
# import seq2seq
# from seq2seq.models import SimpleSeq2Seq
# from seq2seq.models import AttentionSeq2Seq
# from recurrentshop import LSTMCell, RecurrentSequential
# import recurrentshop
# from recurrentshop.cells import *
from keras.models import load_model
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, Dense, RepeatVector, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Titan
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 1080

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# f = fasttext.load_model('./zh/zh.bin')

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
Load data
MFCC_test.shape = (2000, 246, 39)
Option_test.shape = (2000, 4, 13, 300)
Option_test.shape = (2000, 4)
"""

mfcc_test = sys.argv[1]
ans_test = sys.argv[2]

X = np.load(mfcc_test)
# print('X_test shape = ', len(X))
# print('X_test[0] shape = ', len(X[0]))
# print('X_test[0][0] shape = ', len(X[0][0]))


npz = np.zeros((input_length, input_dim))
X_test = []

for e in X:  # 2000
    x = np.copy(npz)
    x[:len(e)] = e
    X_test.append(x)

MFCC_test = np.array(X_test)

# print('X_test shape = ', X_test.shape)

# np.save('/media/Data/Shanboy/final/data/X_test.npy', X_test)

# MFCC_test = np.load('/media/Data/Shanboy/final/data/X_test.npy')
with (open(ans_test, "r")) as openfile:
    Line = openfile.readlines()
Option_test = [x.strip().split(',') for x in Line]


# print('Option_test[0] = ', Option_test[0])

####juice#####

# X_test = np.load(
#     '/media/Data/Shanboy/final/data/MFCC_cnn_test/CNN_testOutput.npy')
# Y_test = np.load('/media/Data/Shanboy/final/data/g_truth/y_test.npy')
# Y_test.resize(2000, 4, 13, 300)
# Y_test = np.transpose(Y_test, [1, 0, 2, 3])  # (4, 2000, 13, 300)
# print('X_test shape = ', X_test.shape)
# print('Y_test shape = ', Y_test.shape)

# text_input_1 = np.array(Y_test[0])
# text_input_2 = np.array(Y_test[1])
# text_input_3 = np.array(Y_test[2])
# text_input_4 = np.array(Y_test[3])

# print('text_input_1 shape = ', text_input_1.shape)  # (2000, 13, 300)
# print('text_input_2 shape = ', text_input_2.shape)  # (2000, 13, 300)
# print('text_input_3 shape = ', text_input_3.shape)  # (2000, 13, 300)
# print('text_input_4 shape = ', text_input_4.shape)  # (2000, 13, 300)

"""
Pretrain model
"""
pretrain_model
model = FastText('./wiki.zh/wiki.zh.bin')
oov = 0
zeros = np.zeros(300)
# print('zeros shape = ', zeros.shape)

Labela = []
for opts in Option_test:  # 2000
    Labelb = []
    for opt in opts:  # 4
        words = opt.split(' ')
        Labelc = []
        for word in words:
            try:
                v = model[word]
                Labelc.append(v)
            except KeyError:
                oov += 1
                continue
        if len(Labelc) != 13:
            for i in range(0, (13 - len(Labelc))):
                Labelc.append(zeros)
        Labelb.append(Labelc)
    Labela.append(Labelb)

Option_test = np.array(Labela)
# print('oov = ', oov)
# print('Option_test shape = ', Option_test.shape)

"""
Tokenizer
No-pretrain word
"""

# from keras.preprocessing.text import Tokenizer
# nb_words = 10000
# tokenizer = Tokenizer(num_words=nb_words)
# tokenizer.fit_on_texts(Texts)
# # Text = tokenizer.texts_to_sequences(Option_test)
# Text = []
# for opt in Option_test:
#     Text.append(tokenizer.texts_to_sequences(opt))
# # print(np.array(Text).shape)

# tmp = []
# npz = np.zeros((13))
# for opts in Text:
#     tmp2 = []
#     for opt in opts:
#         x = np.copy(npz)
#         x[:len(opt)] = opt
#         tmp2.append(x)
#     tmp.append(tmp2)

# Option_test = np.array(tmp)


# print(Option_test.shape)


"""
Load model
"""

model = load_model('./model_hinge_pretrain_GRU.h5')

# model = Sequential()
# model.add(Bidirectional(LSTM(hidden_dim, return_sequences=False),
#                         input_shape=(input_length, input_dim)))
# model.add(Dense(hidden_dim, activation="relu"))
# model.add(RepeatVector(output_length))
# model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
# model.add(TimeDistributed(Dense(output_dim=output_dim, activation="linear")))
# model.compile(loss='hinge', optimizer='adam')

# model = SimpleSeq2Seq(input_shape=(input_length, input_dim), hidden_dim=hidden_dim,
#                       output_length=output_length, output_dim=output_dim, depth=3)

# model = AttentionSeq2Seq(input_shape=(input_length, input_dim), hidden_dim=hidden_dim,
#                          output_length=output_length, output_dim=output_dim, depth=4)

# model.load_weights('./model/model.h5')

"""
Local
let test.csv w2v
"""
# with (open('./data/test.csv', "r")) as openfile:
#     content = openfile.readlines()
# content = [x.strip() for x in content]
# Y_test = np.array(content)

# print(Y_test.shape)

# f = KeyedVectors.load_word2vec_format('./wiki.zh/wiki.zh.bin', binary=True)
# Labels = []
# for i in range(Y_test.shape[0]):
#     options = Y_test[i].split(',')
#     for sentence in options:
#         words = sentence.split(' ')
#         Label = []
#         for word in words:
#             try:
#                 v = f[word]
#                 Label.append(v)
#             except KeyError:
#                 continue
#         Labels.append(Label)

# Y_test = []
# npz = np.zeros((13, 300))
# for e in Labels:
#     x = np.copy(npz)
#     x[:len(e)] = e
#     Y_test.append(x)
# Y_test = np.array(Y_test)
# # print(Y_test.shape)
# Y_test = np.reshape(Y_test, (2000, 4, 13, 300))
# # print(Y_test.shape)
# np.save('./Y_test.npy', Y_test)

"""
predict the test data(A)
-for seq2seq
Y_pred.shape = (2000, 13, 300)
"""
# Y_pred = model.predict(MFCC_test)

# np.save("/media/Data/Shanboy/final/data/Y_pred.npy", Y_pred)


# def loss(y_pred, y_test):
#     return ((y_pred - y_test) ** 2).mean(axis=None)

# Y_pred = np.load("/media/Data/Shanboy/final/data/Y_pred.npy")
# print(Y_pred.shape)


# record = []
# for i in range(Option_test.shape[0]):
#     A = Option_test[i][0]
#     B = Option_test[i][1]
#     C = Option_test[i][2]
#     D = Option_test[i][3]

#     lossList = []
#     lossList.append(loss(A, Y_pred[i]))
#     lossList.append(loss(B, Y_pred[i]))
#     lossList.append(loss(C, Y_pred[i]))
#     lossList.append(loss(D, Y_pred[i]))

#     record.append(lossList)

# print(record)

# ans = []
# for sample in record:
#     ans.append(sample.index(min(sample)))

# print(len(ans))
# print(ans)

"""
predict the test data(B)
-for hinge
MFCC_test.shape = (2000, 246, 39)
Option_test.shape = (2000, 4, 13)
Option_test.shape = (2000, 4, 13, 300)
"""

# # Option_test = np.resize(Option_test, (4, 2000, 13))

# # print('Option_test[0][0][0] = ', Option_test[0][0][0])
# # print('Option_test[0][2][0] = ', Option_test[0][2][0])

Option_test = np.transpose(Option_test, [1, 0, 2, 3])

# # print('Option_test[0][0][0] = ', Option_test[0][0][0])
# # print('Option_test[2][0][0] = ', Option_test[2][0][0])

record = []

for opt in Option_test:
    record.append(model.predict([MFCC_test, opt]))

record = np.array(record).T
# print('record[0] = ', record[0])

ans = []
for buf in record:
    for sample in buf:
        # print(sample)
        # break
        sampleList = sample.tolist()
        ans.append(sampleList.index(max(sampleList)))

# print('ans len = ', len(ans))

###juice###
# ans_porb = model.predict([X_test, text_input_1, text_input_2,
#                           text_input_3, text_input_4])
# ans_classes = ans_porb.argmax(axis=-1)
# print('ans_prob = ', ans_porb)
# print('ans_classes = ', ans_classes)

# for s in ans_classes:
#     print(s)
# ans = ans_classes
"""
Save result
"""

if not os.path.exists('./result'):
    os.makedirs('./result')
result_path = sys.argv[3]

with open(result_path, 'w') as fout:
    print('id,answer', file=fout)
    for i in range(len(ans)):
        print('{},{}'.format(i + 1, ans[i]), file=fout)
