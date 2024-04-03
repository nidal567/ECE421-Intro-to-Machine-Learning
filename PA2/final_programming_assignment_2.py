import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.datasets import load_iris

def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    err=[]
    Weight_matrices=[]
    Weight_matrices.append(np.random.normal(0,0.1,(X_train.shape[1]+1,hidden_layer_sizes[0])))

    for l in range(len(hidden_layer_sizes)-1):
      Weight_matrices.append(np.random.normal(0,0.1,(hidden_layer_sizes[l]+1,hidden_layer_sizes[l+1])))
    
    Weight_matrices.append(np.random.normal(0,0.1,(hidden_layer_sizes[-1]+1,1)))

    N,d=X_train.shape
    ones=np.ones((N,1)) 
    X_train=np.hstack((ones,X_train))

    for epoch in range(epochs):
      N,d=X_train.shape
      jumble=list(range(N))
      np.random.shuffle(jumble)
      error=0
      for index in jumble:
        X,S=forwardPropagation(X_train[index],Weight_matrices) 
        g=backPropagation(X,y_train[index],S,Weight_matrices)
        Weight_matrices=updateWeights(Weight_matrices,g,alpha)
        error=error+errorPerSample(X[-1],y_train[index])
      err.append(error/N)
    return(err,Weight_matrices)

def forwardPropagation(x, weights):
    #Enter implementation here
    S,X=[],[]
    X.append(x) #appending inital values of input to the X list
    for forwpass in range(len(weights)):
        X_next_sample=[]
        if(forwpass == 0):
            X_sample=x
        else:
            X_sample=X_next
        S_sample=np.dot(weights[forwpass].T,X_sample)
        S.append(S_sample)
        for value in S_sample:
            if(len(S_sample) != 1):
                X_next_sample.append(activation(value))
            else:
                X_next_sample.append(outputf(value))
        X_next=np.insert(np.array(X_next_sample),0,1,axis=0)
        X.append(np.array(X_next_sample))
    return X,S

def activation(s):
  if (s>=0):
    return s
  elif (s<0):
    return 0

def derivativeActivation(s):
  if (s>=0):
    return 1
  elif(s<0):
    return 0

def outputf(s):
  x_L = 1/(1+np.exp(-s))
  return x_L

def derivativeOutput (S):
  x_L = np.exp(-S)/(1+np.exp(-S))**2
  return x_L

def errorf(x_L, y):
  if (y==1):
    return(-np.log(x_L))
  elif (y==-1):
    return(-np.log(1-x_L))

def derivativeError(x_L, y):
  if (y==1):
    return(-(1/x_L))
  elif (y==-1):
    return(1/(1-x_L))


def backPropagation(X,y_n,s,weights):
  G=[]
  S_rev=s[::-1]
  X_rev=X[::-1]
  W_rev=weights[::-1]
  DELTA=[]
  for backpass in range(len(weights)):
    if(backpass==0):
      DELTA.append(np.array(np.dot(derivativeError(X_rev[backpass][0],y_n),derivativeOutput(S_rev[backpass][0]))))
    else:
      W_rev[backpass-1] = np.delete(W_rev[backpass-1],0,0)
      DAct=[]
      for val in S_rev[backpass]:
        DAct.append(derivativeActivation(val))
      DAct=np.array(DAct)
      DELTA.append(np.multiply(np.dot(DELTA[backpass-1],W_rev[backpass-1].T),DAct))
    
  #gradient error wrt w
  DELTA_rev=DELTA[::-1]
  g=[]
  for final in range(len(weights)):
    if(final == 0):
      X[final]=X[final].reshape((len(X[final]),1))
      g=np.dot(X[final],DELTA_rev[final])
      G.append(g)
    else:
      X[final]=np.insert(X[final],0,1,axis=0)
      X[final]=X[final].reshape((len(X[final]),1))
      g=np.dot(X[final],DELTA_rev[final])
      G.append(g)
  return G

def updateWeights(Weights,g,alpha):
  nW=[]
  for r in range(len(g)):
    nW.append(Weights[r]-(alpha*g[r]))
  return(nW)

def errorPerSample(X,y_n):
  #print("error",errorf(X,y_n))
  return(errorf(X,y_n))


def pred(x_n,weights):
  X,_=forwardPropagation(x_n, weights)
  if X[3]>= 0.5:
    return 1
  else:
    return -1

def confMatrix(X_train,y_train,w):
  y_pred=[]
  N,d=X_train.shape
  ones=np.ones((N,1)) 
  X_train_new=np.hstack((ones,X_train))

  for i,_ in enumerate(X_train_new):
    y_pred.append(pred(X_train_new[i],w))

  cm=confusion_matrix(y_train, y_pred)
  return(cm)

def plotErr(e,epochs):
    plt.plot(np.arange(0,epochs),e)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.title('error vs epochs')
    plt.show()

def test_SciKit(X_train, X_test, Y_train, Y_test):
  n= MLPClassifier(random_state=1, alpha=0.00001, hidden_layer_sizes=(30,10), solver='adam')
  n.fit(X_train, Y_train)
  m=n.predict(X_test)
  cM=confusion_matrix(Y_test,m)
  return cM

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
  
    for Hidden_L in ([30,10],[5,5],[10,10]):
      err,w=fit_NeuralNetwork(X_train,y_train,1e-2,Hidden_L,100)
      
      plotErr(err,100)
      
      cM=confMatrix(X_test,y_test,w)
      
      sciKit=test_SciKit(X_train, X_test, y_train, y_test) 

      calculate_Acc(X_train, y_train, X_test, y_test,Hidden_L,w)
      
      print("Confusion Matrix is from Part 1a for test is: \n",cM)
      print("Confusion Matrix from Part 1b is:\n",sciKit)

# This is a new function declared to calculate accuracy.
def calculate_Acc(X_train,y_train,X_test,y_test,hidden_layer,w):
  pred1,pred2=[],[]
  N1,_=X_train.shape
  N2,_=X_test.shape
  ones1=np.ones((N1,1)) 
  X_train_new=np.hstack((ones1,X_train))
  ones2=np.ones((N2,1))
  X_test_new=np.hstack((ones2,X_test))
  for value in X_train_new:
    pred1.append(pred(value,w))
  for value in X_test_new:
    pred2.append(pred(value,w))
  print("For hidden layer",hidden_layer)
  print("Train accuracy",accuracy_score(y_train,pred1))
  print("Test accuracy",accuracy_score(y_test,pred2))

test_Part1()