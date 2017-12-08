#!/bin/bash

wget 'https://github.com/Shanboy5566/ML2017FALL/releases/download/0.0.3/embedding.wv.syn0.npy'
wget 'https://github.com/Shanboy5566/ML2017FALL/releases/download/0.0.4/embedding.syn1neg.npy'
wget 'https://github.com/Shanboy5566/ML2017FALL/releases/download/0.0.2/embedding'
wget 'https://github.com/Shanboy5566/ML2017FALL/releases/download/0.0.1/pretrain-75-0.82.hdf5'
python3 hw4_test.py $1 $2