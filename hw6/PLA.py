"""
Normal import
"""
import sys
import os
from skimage import io
import numpy as np

"""
Load training data
X shape =  (415, 1080000)
"""
# X = []
# file_list = os.listdir(sys.argv[1])
# for file_name in file_list:
#     img_path = sys.argv[1] + '/' + file_name
#     X.append(io.imread(img_path).flatten())
# X = np.array(X)
X = []
img_folder = sys.argv[1]
path = os.listdir(img_folder)
# path.sort()

for i in range(0, len(path)):
    img = io.imread(img_folder + '/' + path[i])
    img = img.flatten()
    X.append(img)
X = np.array(X)
# print('X shape = ', X.shape)

"""
1-1
"""
X_mean = np.mean(X, axis=0)
# print('X_mean shape = ', X_mean.shape)
# io.imsave('./mean.jpg', np.reshape(X_mean,
#                                    (600, 600, 3)).astype(np.uint8))

"""
PLA SVD
U shape =  (1080000, 415)
s shape =  (415, 415)
V shape =  (415, 415)
"""

U, s, V = np.linalg.svd((X - X_mean).T, full_matrices=False)
# s = np.diag(s)
# print('U shape = ', U.shape)
# print('s shape = ', s.shape)
# print('V shape = ', V.shape)

# np.save('/media/Data/Shanboy/hw6/Data/U.npy', U)
# np.save('/media/Data/Shanboy/hw6/Data/s.npy', s)
# np.save('/media/Data/Shanboy/hw6/Data/V.npy', V)

# U = np.load('/media/Data/Shanboy/hw6/Data/U.npy')
# s = np.load('/media/Data/Shanboy/hw6/Data/s.npy')
# V = np.load('/media/Data/Shanboy/hw6/Data/V.npy')


"""
1-2
Eigenface shape =  (1080000, 4)
"""
# k = 4

# for i in range(0, k):
#     M = U[:, i]
#     M -= np.min(M)
#     M /= np.max(M)
#     M = (M * 255).astype(np.uint8)
# io.imsave('./' + str(i) + '.jpg',
#           np.reshape(M, (600, 600, 3)).astype(np.uint8))

"""
1-3
"""
# for i in range(0, 4):
#     img = io.imread(img_folder + '/' + str(i) + '.jpg')
#     y = img.flatten()
#     y = y - X_mean
#     w = np.zeros((4,))
#     for s in range(0, 4):
#         w[s] = np.dot(y, U[:, s])
#     reconstruct = np.zeros((600 * 600 * 3,))
#     for j in range(0, 4):
#         reconstruct = reconstruct + (w[j] * U[:, j])
#     M1 = reconstruct + X_mean
#     M1 -= np.min(M1)
#     M1 /= np.max(M1)
#     M1 = (M1 * 255).astype(np.uint8)
#     io.imsave('./reconstruct_' + str(i) + '.jpg',
#               np.reshape(M1, (600, 600, 3)).astype(np.uint8))

"""
For github
"""
img = io.imread(sys.argv[1] + '/' + sys.argv[2])
y = img.flatten()
y = y - X_mean
w = np.zeros((4,))
for s in range(0, 4):
    w[s] = np.dot(y, U[:, s])
reconstruct = np.zeros((600 * 600 * 3,))
for j in range(0, 4):
    reconstruct = reconstruct + (w[j] * U[:, j])
M2 = reconstruct + X_mean
M2 -= np.min(M2)
M2 /= np.max(M2)
M2 = (M2 * 255).astype(np.uint8)
io.imsave('./reconstruction.jpg',
          np.reshape(M2, (600, 600, 3)).astype(np.uint8))


"""
1-4
ratio =  0.0414462483826 = 4.1%
ratio =  0.0294873222511 = 3.0%
ratio =  0.0238771129321 = 2.4%
ratio =  0.022078415569 = 2.2%
"""
# for i in range(0, k):
#     print('ratio = ', s[i] / np.sum(s))
