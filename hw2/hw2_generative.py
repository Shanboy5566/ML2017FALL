import numpy as np
import pandas as pd
import sys, csv

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def normalize(X):
    np.seterr(divide='ignore', invalid='ignore')
    X_normed = ( X - X.mean(axis=0) ) / X.std(axis=0) 
    where_are_NaNs = np.isnan(X_normed)
    X_normed[where_are_NaNs] = 0
    return X_normed

def readData(featDatapath, labelDatapath):
    feature = pd.read_csv(featDatapath) 
    label = pd.read_csv(labelDatapath) 
    
    X = []
    
    for row, col in feature.iterrows():
        x = []
        for v in col[0:]:
            x.append(float(v)) #1次項
#             if v == 1:
#                 x.append(float((2*v)**2)) #2次項
#             else:
#                 x.append(float(v**2))
        X.append(x)
    
    X = np.array(X)
    
    Y = label.values
    Y = Y.reshape(X.shape[0])
    return X, Y

def gen_model(W, b):
    np.save('./modelW.npy',W)
    np.save('./modelB.npy',b)

def read_model(modelWpath, modelbpath):
    W = np.load(modelWpath)
    b = np.load(modelbpath)

def read_test(testpath):
    feature = pd.read_csv(testpath) 
    
    X_TEST = []
    
    for row, col in feature.iterrows():
        x = []
        for v in col[0:]:
            x.append(float(v)) #1次項
#             if v == 1:
#                 x.append(float((2*v)**2)) #2次項
#             else:
#                 x.append(float(v**2))
        X_TEST.append(x)
    
    X_TEST = np.array(X_TEST)
    
    return X_TEST

def gen_ans(anspath, X_TEST, W, b):
    fwb_TEST = sigmoid(np.dot(X_TEST, W) + b)
    Y_TEST = []
    for i in fwb_TEST:
        Y_TEST.append(1) if np.less_equal(i, 0.5) else Y_TEST.append(0)
    with open(anspath, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(Y_TEST)): writer.writerow((i+1, Y_TEST[i]))

if __name__ == '__main__':
    # load pima indians dataset
    X , Y  = readData(sys.argv[3],sys.argv[4])
    X_test = read_test(sys.argv[5])

    # Normalize
    X = normalize(X)
    X_test = normalize(X_test)

    # Generative Process
    x0, x1 = [], []
    for i in range(Y.shape[0]):
        x0.append(X[i]) if Y[i] == 0 else x1.append(X[i])
    x0, x1 = np.array(x0).T , np.array(x1).T
    N0 = x0.shape[1]
    N1 = x1.shape[1]
    u0, u1 = [], []
    for i in range(X.shape[1]):
        u0.append(np.mean(x0[i]))
        u1.append(np.mean(x1[i]))
    u0 = np.array(u0)
    u1 = np.array(u1)

    cov0 = np.cov(x0)
    cov1 = np.cov(x1)
    cov = (cov0*N0 + cov1*N1) / (N0+N1)

    covInv = np.linalg.inv(cov)

    W = np.dot((u0-u1).T , covInv).T

    b = -0.5 * np.dot(np.dot(u0.T,covInv),u0) + 0.5 * np.dot(np.dot(u1.T,covInv),u1) + np.log(N0/N1)

    gen_ans(sys.argv[6],X_test,W,b)