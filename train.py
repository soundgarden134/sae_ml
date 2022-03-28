# Deep-Learning:Training via BP+GD


import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    .....        
    return(w,costo)

# AE's Training 
def train_ae(...):
    W = ut.iniW_ae(...)            
    ....
    return(..)

#SAE's Training 
def train_sae(...):
    ...
    return(W,x) 
   
# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()            
    xe,ye       = ut.load_data_csv('dtrain.csv')   
    W,Xr        = train_sae(xe,p_sae)         
    Ws, cost    = train_softmax(Xr,ye,p_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

