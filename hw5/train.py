"""
Normal import
"""
import numpy as np
import os
import sys
from io import BytesIO

"""
GPU Area
"""
import tensorflow as tf

from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Input, Add, Dot, Concatenate, Dense, Dropout
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
# from keras.regularizers import l2

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Titan
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 1080

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""
Self define
"""


def rmse(yTrain, yPred):
    # yPred = K.clip(yPred, 1, 5)
    return K.sqrt(K.mean(K.square(yPred - yTrain), axis=-1))


"""
Model compile
"""


def gen_model(n_users, n_items, latent_dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])

    user_vec = Embedding(n_users, latent_dim,
                         embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    user_bias = Embedding(
        n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)

    item_vec = Embedding(n_items, latent_dim,
                         embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    item_bias = Embedding(
        n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)

    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])

    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])
    model.summary()
    return model


def nn_model(n_users, n_items, latent_dim=666):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])

    user_vec = Embedding(n_users, latent_dim,
                         embeddings_initializer='uniform')(user_input)
    user_vec = Flatten()(user_vec)

    item_vec = Embedding(n_items, latent_dim,
                         embeddings_initializer='uniform')(item_input)
    item_vec = Flatten()(item_vec)

    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dense(256, activation='relu')(merge_vec)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dropout(0.3)(hidden)
    output = Dense(1)(hidden)

    model = Model([user_input, item_input], output)
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])
    model.summary()
    return model


"""
Load data
X_train shape = (899873, 2)
Y_train shape = (899873, 1)
"""

train_path = './train.csv'
data = np.genfromtxt(train_path, delimiter=",",
                     usecols=(1, 2, 3), skip_header=1)

X_train = data[:, 0:2] - 1
Y_train = data[:, 2]
Y_train = np.resize(Y_train, (899873, 1))

print(X_train)
# print(X_train.shape)
# print(Y_train.shape)

"""
Count the:
the num of users = 6040
the num of movies = 3952
"""

flag1 = -1
flag2 = -1
num_user = 0
num_movie = 0

# for sample in data:
#     if sample[1] != flag1:
#         flag1 = sample[1]
#         num_user += 1
#     if sample[2] != flag2:
#         flag2 = sample[2]
#         num_movie += 1

# input_data['UserID'].drop_duplicates().max()

num_user = int(X_train[:, 0].max()) + 1
num_movie = int(X_train[:, 1].max()) + 1

# print('num of user = ', num_user)
# print('num of movies = ', num_movie)


"""
Rating normalization
"""
Y_std = np.std(Y_train)
Y_m = np.mean(Y_train)
Y_train = (Y_train - Y_m) / Y_std

# print('Y_train normalize = ', Y_train)

"""
Shuffle & split train and validation
"""

split_ratio = 0.10
np.random.seed(66)
indice = np.random.permutation(X_train.shape[0])
X_data = X_train[indice]
Y_data = Y_train[indice]

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)

X_data = X_train[indices]
Y_data = Y_train[indices]

num_validation_sample = int(split_ratio * X_data.shape[0])

X_train = X_data[num_validation_sample:]
Y_train = Y_data[num_validation_sample:]
X_val = X_data[:num_validation_sample]
Y_val = Y_data[:num_validation_sample]

# print('X_train shape = ', X_train.shape)
# print('Y_train shape = ', Y_train.shape)
# print('X_val shape = ', X_val.shape)
# print('Y_val shape = ', Y_val.shape)

"""
Build Model
"""
np_epoch = 100
batch_size = 128

model_path = sys.argv[1]

model = gen_model(num_user, num_movie, latent_dim=666)
# model = nn_model(num_user, num_movie, latent_dim=666)
checkpoint = ModelCheckpoint(filepath=model_path,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')
earlystopping = EarlyStopping(
    monitor='val_loss', patience=3, verbose=1, mode='min')

"""
Train model
"""
model.fit([X_train.T[0], X_train.T[1]], Y_train,
          validation_data=([X_val.T[0].T, X_val.T[1].T], Y_val),
          epochs=np_epoch,
          batch_size=batch_size,
          callbacks=[earlystopping, checkpoint]
          )


# if not os.path.exists('./model'):
#     os.makedirs('./model')
# model.save("./model/model_rmse.h5")
