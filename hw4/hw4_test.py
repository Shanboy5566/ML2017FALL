import numpy as np
import sys
import keras
import gensim
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, LSTM
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

model = load_model('./pretrain-75-0.82.hdf5')
output_path = sys.argv[2]

# import modules & set up logging

word2vec = gensim.models.Word2Vec.load('embedding')

max_len = 40 

tests = []
test_data_path = sys.argv[1]

with open(test_data_path, 'r') as f:
    rows = f.readlines()
    for i,row in enumerate(rows):
        if i == 0:
            continue
        for pivot in range(len(row)):
            if row[pivot] == ',':
                tests.append(row[pivot+1:-1])
                tests[-1] = text_to_word_sequence(tests[-1])
                for idx in range(len(tests[-1])):
                    tests[-1][idx] = word2vec[tests[-1][idx]]
                break

# text = open(test_data_path, 'r')
# rows = text.readlines()

# for i,row in enumerate(rows):
#     if i == 0:
#         continue
#     for pivot in range(len(row)):
#         if row[pivot] == ',':
#             tests.append(row[pivot+1:-1])
#             tests[-1] = text_to_word_sequence(tests[-1])
#             for idx in range(len(tests[-1])):
#                 tests[-1][idx] = word2vec[tests[-1][idx]]
#             break

tests = pad_sequences(tests, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)

result = model.predict(tests, batch_size = 1000)
predict = np.argmax(result, axis=1)

with open(output_path, 'w') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    # f.write('id,label\n')
    for i, v in  enumerate(predict):
        f.write('%d,%d\n' %(i, v))
