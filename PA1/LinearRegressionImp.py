import numpy as np
import pandas as pd
from numpy.linalg import inv, pinv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    #Add implementation here
    d=np.shape(X_train)[0] #dimensions of input training set
    new=np.hstack((np.ones((d,1)), X_train)) # adding column of ones
    #We check if the a square matrix x.transpose*x is a singular matrix or not
    try:
      u=np.dot(np.dot(inv(np.dot(new.T,new)),new.T),y_train)
    except:
      u=np.dot(pinv(new),y_train)
    return u

def mse(X_train,y_train,w):
    #Add implementation here
    y=pred(X_train,w) #predicting output using w
    mser=np.square(y_train-y).mean() #mean squared error
    return mser

def pred(X_train,w):
    #Add implementation here
    d=np.shape(X_train)[0] #dimensions of input training set
    new=np.hstack((np.ones((d,1)), X_train)) #adding column of ones
    pred=np.dot(new,w) #predicting output using w
    return pred

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here
    model=linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred=model.predict(X_test)
    return mean_squared_error(Y_test,Y_pred)

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
