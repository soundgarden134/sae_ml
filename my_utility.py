# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
import csv 

# Initialize weights
def iniW_ae(prev,next):    
    ...
    return(w)
# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# STEP 1: Feed-forward of AE
def forward_ae(w, x):
    z_1 = np.matmul(x,w)
    a_1 = act_sigmoid(z_1)
    return(a_1, z_1)
#Activation function

def act_sigmoid(z):
    return(1/(1+np.exp(-z.astype(float))))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(a_2, x):   
    c_n = 1/2 * np.sum((a_2 - x)**2)
    E = 1/(2*x.shape[1]) * np.sum(c_n)
    return(E, c_n)    


# Update W of the AE
def updW_ae(W, learning_rate, E, xe, A, Z):

    delta_2 = E * deriva_sigmoid(Z[2])
    derivative = np.dot(delta_2.astype(float), np.transpose(A[1].astype(float)))
    derivative = derivative.clip(-1,1)
    W[2] = W[2] - learning_rate*derivative
    
    delta_1 = np.dot(np.transpose(W[2]), delta_2) * deriva_sigmoid(Z[1])
    derivative_2 = np.dot(delta_1, np.transpose(xe))
    derivative_2 = derivative_2.clip(-1,1)
    W[1] = W[1] - learning_rate*derivative_2
    return(W)


# def updW_ae2(W, learning_rate, xe, A, Z):

#     E = np.sqrt((A[2] - A[0]))
#     delta_2 = E * deriva_sigmoid(Z[2])
#     derivative = np.dot(delta_2.astype(float), np.transpose(A[1].astype(float)))
#     W[2] = W[2] - learning_rate*derivative

#     delta_1 = np.dot(np.transpose(W[2]), delta_2) * deriva_sigmoid(Z[1])
#     derivative_2 = np.dot(delta_1, np.transpose(xe))
#     W[1] = W[1] - learning_rate*derivative_2
#     return(W)


def forward_softmax(Xr, W):
    z = np.dot(np.transpose(Xr),W)    
    a = softmax(z)
    return a

# Softmax's gradient 
def grad_softmax(a,ye, Ws, Xr, penalty_rate): 
    num_samples = ye.shape[0]
    num_classes = ye.shape[1]
    cost = np.zeros(num_samples)
    for i in range(num_samples):
        for j in range(num_classes):
            cost[i] += ye[i][j] * np.log(a[i,j]) 
    gW = (-1/num_samples) * np.dot(Xr, ye-a) + penalty_rate* Ws[1] 

    return cost, gW

# Calculate Softmax
def softmax(z):
    exp = np.exp(z)
    return exp/np.sum(exp)


# Feed-forward of the DL
def forward_dl(x, W):
    a_1, z_1 = forward_ae(x, W[1])
    a_2, z_2 = forward_ae(a_1, W[2]) 
    zv = forward_softmax(a_2, W[3])

    
    return(zv)
    

# Métrica
def metricas(yv, zv):
    num_classes = yv.shape[1]
    confusion_matrix = create_confusion_matrix(yv,zv)
    precision = np.zeros(yv.shape[1])
    recall = np.zeros(yv.shape[1])
    f_score = np.zeros(yv.shape[1])
    accuracy = np.zeros(yv.shape[1])
    
    for i in range(yv.shape[1]): #por cada clase
        tp = np.sum(np.diag(confusion_matrix))
        fp = np.sum(np.sum(confusion_matrix, axis = 0)) - tp
        fn = np.sum(np.sum(confusion_matrix, axis = 1)) - tp
        tn = []
        for j in range(num_classes):
            temp = np.delete(confusion_matrix, j, 0)    # delete ith row
            temp = np.delete(temp, j, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        precision[i] = tp/(tp+fp)
        recall[i] = tp/(tp+fn)
        f_score[i] = 2 * (precision[i]*recall[i])/(precision[i] + recall[i])
        accuracy[i] = (tp + tn[i])/(tp + fp + tn[i] + fn)
    print("Precision:" + str(precision))
    print("Recall:" + str(recall))
    print("F score:" + str(f_score))
    print("Accuracy:" + str(accuracy))


    
#Confusion matrix
def create_confusion_matrix(yv, zv):  
    confusion_matrix = np.zeros((yv.shape[1], yv.shape[1]))
    zv = make_prediction(zv)
    for i in range(yv.shape[0]):
        pred = np.argmax(zv[i])
        real_y = np.argmax(yv[i])
        confusion_matrix[pred, real_y] += 1
        
    print(confusion_matrix)
    return confusion_matrix

def make_prediction(zv):

    for i in range(zv.shape[0]):
        max_index = np.argmax(zv[i])
        for j in range(zv.shape[1]):
            if max_index == j:
                zv[i,j] = 1
            else:
                zv[i,j] = 0
    return zv
        
    
#-----------------------------------------------------------------------
# Configuration of the DL 
#-----------------------------------------------------------------------
def load_config():      
    sae_path = 'data/cnf_sae.csv'
    softmax_path = 'data/cnf_softmax.csv'
    
    config_sae = pd.read_csv(sae_path, header=None)
    config_softmax = pd.read_csv(softmax_path, header=None)
    
    return(config_sae, config_softmax)

# Binary Label from raw data 
def Label_binary():
    
    return()

# Load data 
def load_data_csv(path):
    data = pd.read_csv(path,header=None)
    xe = data.iloc[:-1, :]
    ye = data.iloc[-1, :]
    ye = pd.get_dummies(ye) #one hot encoder
    ye = np.array(ye)

    return(xe,ye)

# save weights of both SAE and Costo of Softmax
def save_w_dl(W, Ws, cost):


    np.savez("data/w_dl", w1 = W[1], w2 = W[2], Ws = Ws[1])
    
    np.savetxt('data/mse_softmax.csv', cost, delimiter=',') #costo


    
#load weight of the DL in numpy format
def load_w_dl(filepath):
    
    weights = np.load(filepath)
    W = {1: weights['w1'], 2: weights['w2'], 3: weights['Ws']}
    
    return W

# save weights in numpy format
def save_w_npy():  
    return()