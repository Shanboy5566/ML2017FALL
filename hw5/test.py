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
from keras.models import Model
from keras.layers import Flatten, Input, Add, Dot
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K

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


def draw(x, y):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    y = np.array(y)
    x = np.array(x, dtype=np.float64)

    vis_data = TSNE(n_components=2).fit_transform(x)

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    plt.colorbar(sc)
    plt.savefig('./MF.png')
    # plt.show()


"""
Read the train target
mean and standard
train_std =  1.11689766115
train_m = 3.58171208604
"""
# train_path = './train.csv'
# data = np.genfromtxt(train_path, delimiter=",",
#                      usecols=(1, 2, 3), skip_header=1)

# Y_train = data[:, 2]
# Y_train = np.resize(Y_train, (899873, 1))

# Y_std = np.std(Y_train)
# Y_m = np.mean(Y_train)

# print("Y_std = ", Y_std)
# print("Y_m = ", Y_m)


"""
Load Test Data
"""
train_std = 1.11689766115
train_m = 3.58171208604

test_path = sys.argv[1]
X_test = np.genfromtxt(test_path, delimiter=",", skip_header=1)

"""
Load model & predict
"""

model = load_model('./nn_model.h5', custom_objects={'rmse': rmse})

tmp_pred = model.predict([X_test.T[1] - 1, X_test.T[2] - 1])
pred = (tmp_pred * train_std) + train_m
# pred = tmp_pred

"""
Save result
"""

result_path = sys.argv[2]

with open(result_path, 'w') as file:
    file.write("TestDataID,Rating\n")
    for i in range(np.shape(X_test)[0]):
        # if pred[i] > 5.0:
        #     pred[i] = 5.0
        # if pred[i] < 1.0:
        #     pred[i] = 1.0
        file.write("%d,%.14f\n" % (X_test[i][0], pred[i]))


"""
Get movie embedding
"""
# user_emb = np.array(model.layers[2].get_weights()).squeeze()
# print('user embedding shape:', user_emb.shape)
# movie_emb = np.array(model.layers[3].get_weights()).squeeze()
# print('movie embedding shape:', movie_emb)

# np.save('./user_emb.npy', user_emb)
# np.save('./movie_emb.npy', movie_emb)

# draw(user_emb, movie_emb)
