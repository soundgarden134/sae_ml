# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

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
    W[2] = W[2] - learning_rate*derivative
    
    delta_1 = np.dot(np.transpose(W[2]), delta_2) * deriva_sigmoid(Z[1])
    derivative_2 = np.dot(delta_1, np.transpose(xe))
    W[1] = W[1] - learning_rate*derivative_2
    return(W)

def updW_ae(w1,u,T,a1,a2):
    #decoder
    g = np.cross((a1 - a2),deriva_sigmoid(np.dot(w1,a1)))
    d = np.dot(g,np.power(a1,T))
    w2 = w1  - np.dot(u,d)
    return(w2)


# Softmax's gradient 
def grad_softmax(xe,y,w,lambW, A, learning_rate):    
    Cost = (1/xe.shape[1])*np.sum(np.sum(ye*np.log(A) + lambW/2 * np.linalg.norm(w)))
    derivative = (-1/xe.shape[1])*((y-A) + np.transpose(xe))+lambW*w 
    gW = w - learning_rate*derivative
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
    a_n = z/np.sum(z)
    return()

# Feed-forward of the DL
def forward_dl(a,w_2):        
  
    z_2 = np.matmul(np.transpose(a),np.transpose(w_2))
    a_2 = act_sigmoid(z_2)
    
    return(a_2)
    

# Métrica
def metricas(c_m):
    N = ye.shape[0]
    precision = c_m/np.sum(c_m)
    recall = c_m/(np.sum(c_m))
    F_score = 2* ((precision * recall)/(precision + recall))
    accuracy = (1/N)*np.sum(c_m)
    return()
    
#Confusion matrix
def confusion_matrix(A, ye):  
    
    return()
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

    return(xe,ye)

# save weights of both SAE and Costo of Softmax
def save_w_dl():    
    return()
        
    
#load weight of the DL in numpy format
def load_w_dl():

    return()    

# save weights in numpy format
def save_w_npy():  
    return()