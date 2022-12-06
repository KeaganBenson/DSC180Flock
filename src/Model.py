from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
def Train_model(trainDF,x,y):
    xTrain = trainDF[x].to_numpy()
    yTrain = trainDF[y].to_numpy()
    reg = LinearRegression().fit(xTrain, yTrain)
    return reg

def PrepareTestSet(testDF, model,x,name):
    testDF[name] = np.round(model.predict(testDF[x].to_numpy()))
    return testDF.reset_index(drop = 'True')