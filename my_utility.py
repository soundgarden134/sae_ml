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
    z_1 = np.matmul(x,np.transpose(w))
    a_1 = act_sigmoid(z_1)
    return(a_1)
#Activation function

def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(a_2, x):   
    c_n = 1/2 * np.sum((a_2 - x)**2)
    return()    

# Update W of the AE
def updW_ae(w, learning_rate, error, xe):
    W = 0  
    return(W)

# Softmax's gradient
def grad_softmax(x,y,w,lambW):    

    return(gW,Cost)

# Calculate Softmax
def softmax(z):

        return(...)

# Feed-forward of the DL
def forward_dl(a,w_2):        
  
    z_2 = np.matmul(np.transpose(a),np.transpose(w_2))
    a_2 = act_sigmoid(z_2)
    
    return(a_2)
    


# Métrica
def metricas():
  
    return()
    
#Confusion matrix
def confusion_matrix():    
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
    data = pd.read_csv(path, header=None)
    data = np.transpose(data)
    xe = data.iloc[:, :-1]
    ye = data.iloc[:, -1]


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