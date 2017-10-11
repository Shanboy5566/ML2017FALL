# coding: utf-8

import csv
import sys
import numpy as np
import scipy as sp
import pandas as pd

# 1. read model
W_BEST = np.load('modelWbest.npy')
B_BEST = np.load('modelBbest.npy')

# ### 2. Setup Environment Parameters
hr_len = 9

# ### 3. Read Testing Data

test = pd.read_csv(sys.argv[1] , header=None , encoding='Big5')

test_format = [[] for i in range(18)]
t_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # 18 feature

for row , col in test.iterrows():
    if row%18 in t_list:
#         print(row%18)
        for value in col[2:]:
            if value == 'NR':
                value = 0
            test_format[row%18].append(float(value)) 
    else:
        for value in col[2:]:
            if value == 'NR':
                value = 0
            test_format[row%18].append(0) 

test_format = np.array(test_format)

#         test_format.shape

row, col = np.shape(test)
num_test = int(row / 18)

#         num_test

# ### 4. Create Testing Dataset
X_TEST = []
for i in range(num_test):
    x = []
    for type_id in t_list:
        for hr in range(hr_len):
            x.append(test_format[type_id][i*hr_len+hr])
            x.append(test_format[type_id][i*hr_len+hr] **2) #2次項
    X_TEST.append(x)
X_TEST = np.array(X_TEST)

# ### 5. Calculate for Predicted PM2.5 & Generate CSV output
WX_TEST = np.dot(X_TEST, W_BEST)
Y_TEST = WX_TEST + B_BEST

with open(sys.argv[2], 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['id', 'value'])
    for i in range(num_test): writer.writerow(('id_{0}'.format(i), Y_TEST[i]))