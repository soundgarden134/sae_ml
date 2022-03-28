# Deep-Learning:Training via BP+GD
import pandas     as pd
import numpy      as np
import my_utility as ut
import math
	
# Softmax's training
def train_softmax(x,y,param):  
    w = 1
    costo = 1    
    return(w,costo)

# AE's Training 
def train_ae():
    W = ut.iniW_ae()            
    return()

#SAE's Training 
def train_sae(xe, p_sae):
    

    n0 = xe.shape[0]
    n1 = p_sae.iloc[2].values[0]
    r = math.sqrt(6/(n1+n0))
    w = []
    w[0] = np.random.rand(n1,n0)*2*r-r
    w[1] = np.random.rand(n1,n0)*2*r-r

    return(n1,n0) 
   
# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()            
    xe,ye = ut.load_data_csv('data/dtrain.csv')   
    W,Xr = train_sae(xe,p_sae)         
    # Ws, cost    = train_softmax(Xr,ye,p_sft)
    # ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

