# coding: utf-8

import csv

# ### 1. Read Training Data
import numpy as np
import scipy as sp
import pandas as pd

data = pd.read_csv('./train.csv',encoding='Big5')
row, col = np.shape(data) # Row = 4320 , Col = 27

# t_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17] # 
# t_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12] # 10052316
t_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # 18 feature
# t_list = [9] # Only PM2.5

data_format = [[] for i in range(18)]

for row , col in data.iterrows():
    if row%18 in t_list:
#         print(row%18)
        for value in col[3:]:
            if value == 'NR':
                value = 0
            data_format[row%18].append(float(value)) 
    else:
        for value in col[3:]:
            data_format[row%18].append(0) 

data_format = np.array(data_format)

data_format.shape

# ### 2. Setup Environment Parameters
fold = 12 # for N-Fold Cross Validation
num_type = len(data)
# hr_len = [9 , 5] #To predict the PM2.5 will be related to 'hr_len' hour before
# Lambda_list = [0.1 , 0.01 , 0.001 , 0.0001]
bestRMSE = []
hr_len = [9] #To predict the PM2.5 will be related to 'hr_len' hour before
Lambda_list = [0.1]
# hr_len = 9

for hr_len in hr_len:
    for Lambda in Lambda_list:
        # ### 3. Generate All Samples
        X_ALL = []
        Y_ALL = []
        for month in range(12):
            for hour_start in range(471):
                x = []
                hour_predict = hour_start + hr_len
                for type_id in t_list:
                    for hr in range(hour_predict-hr_len, hour_predict):
                        x.append(data_format[type_id][month*480+hr]) #Take 'hr_len' hour before data to feature
                        x.append(data_format[type_id][month*480+hr] ** 2) # 2次項
                Y_ALL.append(data_format[9][month*480+hour_predict]) #Ground truth with PM2.5
                X_ALL.append(x)

        # shuffle all the samples
        XY_ALL = list(zip(X_ALL, Y_ALL))
        np.random.shuffle(XY_ALL)
        X_ALL, Y_ALL = zip(*XY_ALL)
        X_ALL = np.array(X_ALL)
        Y_ALL = np.array(Y_ALL)

        #         X_ALL

        #         print('X_ALL = ' , X_ALL.shape )
        #         print('Y_ALL = ' , Y_ALL.shape )

        # ### 4. Create Training & Validation Datasets + Calculate w & b + RMSE
        RMSE_BEST = 100
        W_BEST = None
        B_BEST = None

        for cross_index in range(fold):
            X_TRAIN = []
            Y_TRAIN = []
            X_VALIDATION = []
            Y_VALIDATION = []
            for i in range(X_ALL.shape[0]):
                if cross_index*471 <= i and i < (cross_index+1)*471:
                    X_VALIDATION.append(X_ALL[i])
                    Y_VALIDATION.append(Y_ALL[i])
                else:
                    X_TRAIN.append(X_ALL[i])
                    Y_TRAIN.append(Y_ALL[i])
            X_TRAIN = np.array(X_TRAIN)
            Y_TRAIN = np.array(Y_TRAIN)
            X_VALIDATION = np.array(X_VALIDATION)
            Y_VALIDATION = np.array(Y_VALIDATION)

        #             print('X_TRAIN = ' , X_TRAIN.shape)


            W = np.zeros(X_TRAIN.shape[1])

            b = 0
            SUM_SQDW = np.zeros(X_TRAIN.shape[1])
            SUM_SQDB = 0.
            ada_alpha = 1.
            nor_alpha = 0.000000000000005
#             Lambda = 0.1
            adam_alpha = 0.001
            beta1 = 0.9
            beta2 = 0.999
            Wmt = 0
            Wvt = 0
            Bmt = 0
            Bvt = 0
            t = 0
            eps = 1e-8
            epoch = 50000
        #     epoch = 1000



            for i in range(epoch): # iteration (default = 50000)
                t += 1 # time step for admagrad
        #         print('Xtrain shape' , X_TRAIN.shape)
        #         print('W shape' , W.shape)
                WX_TRAIN = np.dot(X_TRAIN, W) # inner product of weight & data
                ERR = Y_TRAIN - (b + WX_TRAIN) # error of predicted result (y-(b+wx))
        #         print('ERR shape' , ERR.shape)
        #         X_TRAIN_T = X_TRAIN.T # transpose data for next inner product
        #         DW = -2 * np.dot(X_TRAIN_T, ERR) # multiply -X to error formula
                DW = -2 * np.dot(ERR, X_TRAIN) # multiply -X to error formula
        #         print('DW shape' , DW.shape)
                DB = -2 * np.sum(ERR) # sum the error

                # Compute Loss & Print
                L = np.sum(ERR ** 2)
                if i%100 == 0 :
                    print ("Hr_len %d |Lambda %.4f |Fold %s | Epoch %s | Loss: %.7f" % (hr_len,Lambda,cross_index ,i, L))

                # Regularization
                DW += Lambda * 2 * W

                # Normal
#                 W = W - nor_alpha * DW # / X_TRAIN.shape[0]
#                 b = b - nor_alpha * DB # / X_TRAIN.shape[0]

                # Adagrad
#                 SUM_SQDW += np.square(DW)
#                 SUM_SQDB += np.square(DB)
#                 W = W - ada_alpha/np.sqrt(SUM_SQDW) * DW # / X_TRAIN.shape[0]
#                 b = b - ada_alpha/np.sqrt(SUM_SQDB) * DB # / X_TRAIN.shape[0]

                # Adamgrad
                Wmt = beta1 * Wmt + (1-beta1) * DW
                Wvt = beta2 * Wvt + (1-beta2) * np.square(DW)
                Wmthat = Wmt / (1-np.power(beta1, t))
                Wvthat = Wvt / (1-np.power(beta2, t))
                Bmt = beta1 * Bmt + (1-beta1) * DB
                Bvt = beta2 * Bvt + (1-beta2) * np.square(DB)
                Bmthat = Bmt / (1-np.power(beta1, t))
                Bvthat = Bvt / (1-np.power(beta2, t))
                W = W - (adam_alpha*Wmthat) / (np.sqrt(Wvthat) + eps)
                b = b - (adam_alpha*Bmthat) / (np.sqrt(Bvthat) + eps)

        #     WX_TRAIN = np.dot(X_TRAIN, W)
            SUMSQERR = L
            RMSE_TRAIN = np.sqrt(SUMSQERR/X_TRAIN.shape[0])

            WX_VALIDATION = np.dot(X_VALIDATION, W)
            SUMSQERR = np.sum((Y_VALIDATION - (b + WX_VALIDATION)) ** 2)
            RMSE_VALIDATION = np.sqrt(SUMSQERR/X_VALIDATION.shape[0])
        #     RMSE = (RMSE_TRAIN + RMSE_VALIDATION) / 2
            RMSE = RMSE_VALIDATION
            if RMSE < RMSE_BEST:
                RMSE_BEST = RMSE
                W_BEST = W
                B_BEST = b
                print ('RMSE_Train: %.7lf' % (RMSE_TRAIN))
                print ('RMSE_Validation: %.7lf' % (RMSE_VALIDATION))
                print ('RMSE: %.7lf' % (RMSE))

            print ('Best RMSE: %.7lf' % RMSE_BEST)
            print ('Best B: %.7lf' % B_BEST)
            print ("Best W:\n%s" % W_BEST)
            print ('Best W shape:\n%s' % W.shape)
            # save model
            np.save('modelW.npy',W_BEST)
            np.save('modelB.npy',B_BEST)
            # read model
#             w = np.load('model.npy')