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
def forward_ae(...):
    ...         
    return(a) 

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(...):    
    ...
    return()    

# Update W of the AE
def updW_ae(...):
    ...
    return(W)

# Softmax's gradient
def grad_softmax(x,y,w,lambW):    
    ...    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
        ....
        return(...)

# Feed-forward of the DL
def forward_dl(x,W):        
    ...    
    return(zv)
    


# Métrica
def metricas(...):
    ...    
    return()
    
#Confusion matrix
def confusion_matrix():    
    return()
#-----------------------------------------------------------------------
# Configuration of the DL 
#-----------------------------------------------------------------------
def load_config():      
    ...
    return(...)

# Binary Label from raw data 
def Label_binary():
    ...  
    return()

# Load data 
def load_data_csv():
    ...  
    return(...)

# save weights of both SAE and Costo of Softmax
def save_w_dl():    
    ....
        
    
#load weight of the DL in numpy format
def load_w_dl():
    ...
    return()    

# save weights in numpy format
def save_w_npy():  
    ....
