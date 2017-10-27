import numpy as np
import pandas as pd
import sys, csv

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy(fx_n, y_n):
    ans = []
    for i in range(fx_n.shape[0]):
        if np.isclose(fx_n[i], 0.): 
            ans.append(y_n[i] * np.log(fx_n[i]+1e-3) + (1-y_n[i]) * np.log(1-fx_n[i]))
        elif np.isclose(1-fx_n[i], 0.):
            ans.append(y_n[i] * np.log(fx_n[i]) + (1-y_n[i]) * np.log(1-fx_n[i]+1e-3))
        else:
            ans.append(y_n[i] * np.log(fx_n[i]) + (1-y_n[i]) * np.log(1-fx_n[i]))
    return -np.array(ans)

def normalize(X):
    np.seterr(divide='ignore', invalid='ignore')
    X_normed = ( X - X.mean(axis=0) ) / X.std(axis=0) 
    where_are_NaNs = np.isnan(X_normed)
    X_normed[where_are_NaNs] = 0
    return X_normed

def readData(featDatapath, labelDatapath, featurePower):
    feature = pd.read_csv(featDatapath) 
    label = pd.read_csv(labelDatapath) 
    
    X = []
    
    for row, col in feature.iterrows():
        x = []
        for v in col[0:]:
            x.append(float(v)) #1次項
            if featurePower >= 2:
                if v == 1:
                    x.append(float((2*v)**2)) #2次項
                else:
                    x.append(float(v**2))
            if featurePower >= 3:
                if v == 1:
                    x.append(float((2*v)**3)) #3次項
                else:
                    x.append(float(v**3))
        X.append(x)
    
    X = np.array(X)
    
    Y = label.values
    Y = Y.reshape(X.shape[0])
    return X, Y

def gen_model(W, b,Wpath ,bpath):
    np.save(Wpath,W)
    np.save(bpath,b)

def read_model(modelWpath, modelbpath):
    W = np.load(modelWpath)
    b = np.load(modelbpath)
    return W, b

def read_test(testpath,featurePower):
    feature = pd.read_csv(testpath) 
    
    X_TEST = []
    
    for row, col in feature.iterrows():
        x = []
        for v in col[0:]:
            x.append(float(v)) #1次項
            if featurePower >= 2:
                if v == 1:
                    x.append(float((2*v)**2)) #2次項
                else:
                    x.append(float(v**2))
            if featurePower >= 3:
                if v == 1:
                    x.append(float((2*v)**3)) #3次項
                else:
                    x.append(float(v**3))
        X_TEST.append(x)
    
    X_TEST = np.array(X_TEST)
    
    return X_TEST

def gen_ans(anspath, X_TEST, W, b):
    # X_TEST = map(float,X_TEST)
    # W      = map(float,W)
    fwb_TEST = sigmoid(np.dot(X_TEST, W) + b)
    Y_TEST = []
    for i in fwb_TEST:
        Y_TEST.append(0) if np.less_equal(i, 0.5) else Y_TEST.append(1)
    with open(anspath, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(Y_TEST)): writer.writerow((i+1, Y_TEST[i]))

def logistic_regression(X, Y, Regularization):
    lossBest = 1000000
    W_BEST = None
    B_BEST = None
    fold = 5
    for cross_index in range(fold): #5-fold
        X_TRAIN = []
        Y_TRAIN = []
        X_VALIDATION = []
        Y_VALIDATION = []

        foldSize = int(X.shape[0] / fold)

        for i in range(X.shape[0]):
            if cross_index == (fold-1):
                if cross_index*(foldSize) <= i and i < (X.shape[0]+1):
                    X_VALIDATION.append(X[i])
                    Y_VALIDATION.append(Y[i])
                else:
                    X_TRAIN.append(X[i])
                    Y_TRAIN.append(Y[i])
            else:
                if cross_index*(foldSize) <= i and i < (cross_index+1)*(foldSize):
                    X_VALIDATION.append(X[i])
                    Y_VALIDATION.append(Y[i])
                else:
                    X_TRAIN.append(X[i])
                    Y_TRAIN.append(Y[i])
        X_TRAIN = np.array(X_TRAIN)
        Y_TRAIN = np.array(Y_TRAIN)
        X_VALIDATION = np.array(X_VALIDATION)
        Y_VALIDATION = np.array(Y_VALIDATION)
#         print('X_TRAIN shape = ' , X_TRAIN.shape)
#         print('Y_TRAIN shape = ' , Y_TRAIN.shape)
#         print('X_VALIDATION shape = ' , X_VALIDATION.shape)
#         print('Y_VALIDATION shape = ' , Y_VALIDATION.shape)

        W, b = np.zeros(X_TRAIN.shape[1]), 0
        SUM_SQDW, SUM_SQDB = np.zeros(X_TRAIN.shape[1]), 0
        norm, adag, adam = 1e-25, 1e-16, 1e-4 # adam-default = 0.001
        beta1, beta2 = 0.9, 0.999
        Wmt, Wvt = 0, 0
        Bmt, Bvt = 0, 0
        epoch, Lambda, t, eps = 10000, Regularization, 0, 1e-8

        for i in range(epoch):
    #         fwb = 1 / (1 + np.exp(-(np.dot(X_TRAIN, W) + b)))
            fwb = sigmoid(np.dot(X_TRAIN, W) + b)
    #         print('fwb shape = ' , fwb.shape , fwb)
            ERR = Y_TRAIN - fwb
    #         print('ERR shape = ' , ERR.shape , ERR)
            DW =  -1 * np.dot(ERR,X_TRAIN)
    #         print('DW shape = ' , DW.shape , DW)
            DB =  -1 * np.sum(ERR)
    #         print('DB shape = ' , DB.shape , DB)

            # Compute Loss & Print
            # if i % 500 == 0:
            #     Loss = np.sum(cross_entropy(fwb, Y_TRAIN))
            #     print ("Fold %s |Iter %7s | Loss: %.7f" % (cross_index,i, Loss))
    #             print('Fold %s | Iter %s' %(cross_index,i))
            # Regularization
    #         print(W.shape)
    #         print(DW.shape)
    #         print(Lambda)
            DW += Lambda * 2 * W

            # Normal
    #         W -= norm * DW # / X_TRAIN.shape[0]
    #         b -= norm * DB # / X_TRAIN.shape[0]

            # Adagrad
    #         SUM_SQDW += np.square(DW)
    #         SUM_SQDB += np.square(DB)
    #         W += adag / np.sqrt(SUM_SQDW) * DW # / X_TRAIN.shape[0]
    #         b += adag / np.sqrt(SUM_SQDB) * DB # / X_TRAIN.shape[0]

            # Adamgrad
            t += 1
            Wmt = beta1 * Wmt + (1-beta1) * DW
            Wvt = beta2 * Wvt + (1-beta2) * np.square(DW)
            Wmthat = Wmt / (1-np.power(beta1, t))
            Wvthat = Wvt / (1-np.power(beta2, t))
            Bmt = beta1 * Bmt + (1-beta1) * DB
            Bvt = beta2 * Bvt + (1-beta2) * np.square(DB)
            Bmthat = Bmt / (1-np.power(beta1, t))
            Bvthat = Bvt / (1-np.power(beta2, t))
            W -= (adam*Wmthat) / (np.sqrt(Wvthat) + eps)
            b -= (adam*Bmthat) / (np.sqrt(Bvthat) + eps)
            WX_TRAIN = np.dot(X_TRAIN, W)
        
        fwbTrain = sigmoid(np.dot(X_TRAIN, W) + b)
        lossTrain = np.sum(cross_entropy(fwbTrain, Y_TRAIN))

        fwbValidation = sigmoid(np.dot(X_VALIDATION, W) + b)
        lossValidation = np.sum(cross_entropy(fwbValidation, Y_VALIDATION))
        loss = lossValidation
        if loss < lossBest:
            lossBest = loss
            W_BEST = W
            B_BEST = b
            # print ('Loss_Train: %.7lf' % (lossTrain))
            # print ('Loss_Validation: %.7lf' % (lossValidation))

#         print('fwbTrain shape = ' , fwbTrain.shape)
#         print('Y_Train shape = ' , Y_TRAIN.shape)
#         print('fwbValidation = ', fwbValidation)
        trainAcc = sum(1 for x,y in zip(fwbTrain,Y_TRAIN) if ( x>0.5 and y==1) or (x<0.5 and y==0) ) / float(len(fwbTrain))
        validationAcc = sum(1 for x,y in zip(fwbValidation,Y_VALIDATION) if ( x>0.5 and y==1) or (x<0.5 and y==0) ) / float(len(fwbValidation))
        # print('Train Accuracy = %.7lf' % trainAcc)
        # print('Validation Accuracy = %.7lf' % validationAcc)
        # print ('Best Loss: %.7lf' % lossBest)
        # print ('Best B: %.7lf' % B_BEST)
        # print ("Best W:\n%s" % W_BEST)
    return W_BEST, B_BEST

if __name__ == '__main__':
    X , Y  = readData(sys.argv[3],sys.argv[4],2)
    X_test = read_test(sys.argv[5],2)

    X = normalize(X)
    X_test = normalize(X_test)

    # W , b = logistic_regression(X,Y,reg)
    W , b = read_model(sys.argv[7],sys.argv[8])
    gen_ans(sys.argv[6],X_test,W,b)


