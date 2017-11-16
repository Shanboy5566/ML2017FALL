#!/bin/sh

wget 'https://github.com/Shanboy5566/ML2017FALL/releases/download/0.0.0/model.h5'
python3 cnn_test.py $1 $2