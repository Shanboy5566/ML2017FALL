# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import sys, csv

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
            if v == 1:
                x.append(float((2*v)**2)) #2次項
            else:
                x.append(float(v**2))
        X.append(x)
    
    X = np.array(X)
    
    Y = label.values
    Y = Y.reshape(X.shape[0])
    return X, Y

def read_test(testpath):
    feature = pd.read_csv(testpath) 
    
    X_TEST = []
    
    for row, col in feature.iterrows():
        x = []
        for v in col[0:]:
            x.append(float(v)) #1次項
            if v == 1:
                x.append(float((2*v)**2)) #2次項
            else:
                x.append(float(v**2))
        X_TEST.append(x)
    
    X_TEST = np.array(X_TEST)
    
    return X_TEST


def keras_ann(X, Y):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # define N-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []

    for train, test in kfold.split(X, Y):
        # create model
        model = Sequential()
        model.add(Dropout(0.2, input_shape=(212,)))
        model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X[train], Y[train], epochs=20, batch_size=10, verbose=1)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    return model


if __name__ == '__main__':
    # load pima indians dataset
    X , Y  = readData(sys.argv[3],sys.argv[4])
    X_test = read_test(sys.argv[5])

    # Normalize
    X = normalize(X)
    X_test = normalize(X_test)

    # Train ann model
    # model = keras_ann(X, Y)

    # Load model
    model = load_model(sys.argv[7])

    # calculate predictions
    predictions = model.predict(X_test)

    # round predictions
    rounded = [int(round(x[0])) for x in predictions]
    # print(rounded)

    with open(sys.argv[6], 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(rounded)): writer.writerow((i+1, rounded[i]))




