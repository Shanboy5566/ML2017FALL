"""
Normal import
"""
import os
import numpy as np
import sys
# from matplotlib import pyplot as plt
"""
GPU Area
"""
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Titan
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 1080

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""
Load image
Image shape =  (140000, 784)
"""
image = np.load(sys.argv[1])
# image = image / 255.
# print('Image shape = ', image.shape)

"""
PCA
newImage shape =  (140000, dim)
"""
# dim = 512
# pca = PCA(n_components=dim)
# newImage = pca.fit_transform(image)
# print('newImage shape = ', newImage.shape)

"""
t-SNE
"""
# dim = 128
# newImage = TSNE(n_components=2).fit_transform(image)
# print('newImage shape = ', newImage.shape)

"""
Auto encoder
"""
# model_path = sys.argv[1]

# input_img = Input(shape=(784,))
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(32, activation='relu')(encoded)

# decoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(128, activation='relu')(decoded)
# decoded = Dense(784, activation='relu')(decoded)

# autoencoder = Model(input_img, decoded)
# # this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)

# autoencoder.compile(optimizer='adadelta', loss='mse')

# earlystopping = EarlyStopping(
#     monitor='val_loss', patience=2, verbose=1, mode='auto')
# checkpoint = ModelCheckpoint(filepath=model_path,
#                              verbose=1,
#                              save_best_only=True,
#                              monitor='val_loss',
#                              mode='min')

# autoencoder.fit(image, image,
#                 epochs=100,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_split=0.1, callbacks=[checkpoint])

# encoder.save('./model/encoder.h5')


"""
Clustering
"""
# encoder = load_model('./model/encoder.h5')
encoder = load_model('./encoder.h5')
encoded_imgs = encoder.predict(image)

# print('encoded_imgs shape = ', encoded_imgs.shape)

kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
# print('newImage clustering = ', kmeans.labels_)


"""
Load test data
test data shape =  (1980000, 3)
"""
test = np.genfromtxt(sys.argv[2],
                     delimiter=',', skip_header=1)
# print('test data shape = ', test.shape)

ans = []
for data in test:
    ID = int(data[0])
    index_1 = int(data[1])
    index_2 = int(data[2])

    if kmeans.labels_[index_1] == kmeans.labels_[index_2]:
        ans.append(1)
    else:
        ans.append(0)

# if not os.path.exists('./result'):
#     os.makedirs('./result')

result_path = sys.argv[3]

with open(result_path, 'w') as fout:
    print('ID,Ans', file=fout)
    for i in range(len(ans)):
        print('{},{}'.format(i, ans[i]), file=fout)
