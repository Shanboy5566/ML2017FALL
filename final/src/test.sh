#!/bin/bash

wget 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.zip'
unzip wiki.zh.zip

python test.py $1 $2 $3