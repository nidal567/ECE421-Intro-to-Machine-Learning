import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix,accuracy_score


def fit_perceptron(X_train, y_train):
    #Add implementation here 
    max_epochs=5000                             #Intializing the total number of epochs to run
    N,d=X_train.shape                           # Takes the shape of the X data matrix, in this case N sample each with d attributes
    ones=np.ones((N,1)) 
    X_train=np.hstack((ones,X_train))           # We add a column of ones to them to each of the same as we add a new X0 term for our W0 (W0==-B)
    Weights=np.zeros(X_train.shape[1])          # The weight vector are of dimension d+1 
    for epoch in range(max_epochs):
        #print(epoch)
        Value=errorPer(X_train, y_train, Weights) # Calls the errorPer to calculate our loss so that we know when to stop
        #print(Value)                             #As this is the pocket algorithm and it stops when the 
        if(epoch==0):                             #E(in) of W+1 is greater than E(in) of W
            Prev_Loss=Value
        else:
            if(round(Value,2)==0.00):            #The else condition checks for E(in) and based on that stops the execution
                return Weights                   #It returns the final weights that can be used on Test data
                break
            else:
                Prev_Loss=Value                #It would reach this else state only when there is an update of weights
                prev_weights=Weights           #Stores the previous weights just incase the above else condition is true
                Weights=Weights+y_train[vector]*X_train[vector]
    return prev_weights                         #The final weights are returned 

def errorPer(X_train,y_train,w):
    #Add implementation here 
    y_predicted=[]                              #This function is for calcualting the Loss and also providing the index     
    for i in range(y_train.shape[0]):           #of the which sample of X needs to taken to update the weights
        y_predicted.append(pred(X_train[i],w))
    y_predicted=np.array(y_predicted)    
    global vector  
    vector=np.argmax((y_predicted!=y_train)==True)     
    return ((1*(y_predicted!=y_train)).sum())/y_train.shape[0]

def confMatrix(X_train,y_train,w):
    #Add implementation here
    ConfusionMatrix=np.array([[0,0],[0,0]])     #In this function we calculate the Confusion Matrix
    N,d=X_train.shape                           #It is a matrix that can depict how many samples were correctly predicted
    ones=np.ones((N,1))                         #and how many were incorrectly predicted. Its a 2x2 matrix with the correctly
    X_train=np.hstack((ones,X_train))           #predicted samples depicted as the primary diagonals of the matrix.
    output=[]
    for sample in range(y_train.shape[0]):
        prediction=pred(X_train[sample],w)
        output.append(prediction)
        if(prediction == y_train[sample]):
            if prediction == -1:
                ConfusionMatrix[0,0]+=1
            elif prediction == 1:
                ConfusionMatrix[1,1]+=1
        else:
            if prediction == -1:
                ConfusionMatrix[0,1]+=1
            elif prediction == 1:
                ConfusionMatrix[1,0]+=1 
    print('Precision score',accuracy_score(output, y_train))
    return ConfusionMatrix                     #We return the calcualted confusion matrix

def pred(X_train,w):
    #Add implementation here
    y_pred=np.dot(X_train,w)                    #We use this function to predict the output. Its based of the h=sign(X.T*W)
    #print(y_pred)              
    if y_pred > 0:                              #The if elif conditions are used to calculate the sign function
        return(+1)
    elif y_pred < 0:
        return(-1)
    else:
        return(y_pred)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here 
    #We set the max iteration to 5000 and the exit criteria as 0.0001
    percp= Perceptron(max_iter=5000, tol=1e-3, random_state=0)#,verbose=1) 
    percp.fit(X_train,Y_train) #We fit the model to X_train
    print('Precision score using sciKit',percp.score(X_train,Y_train))
    y_pred= percp.predict(X_test) #The fit model is used to predict X_test data
    return confusion_matrix(Y_test,y_pred)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
 
'''
ANSWERS:
1. tol pararmeter is the stopping criterion. It will stop when loss>previous_loss-tol
2. max_iter= 5000 doesnt not guarentee that the data will be passed pver 5000 times.
3. We set the weights of a model by calcualting it using the least square solution.
4. We can judge that using the accuracy score.
'''