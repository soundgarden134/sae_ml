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
def train_ae(xe, p_sae):

    n0 = int(xe.shape[0])
    n1 = int(p_sae.iloc[2].values[0])
    n2 = int(p_sae.iloc[3].values[0])
    num_iter = int(p_sae.iloc[0].values[0])
    learning_rate = p_sae.iloc[1].values[0]
    
    W = {}
    W = {1: ut.iniW(n1, n0), 2: ut.iniW(n1,n0)}
    W[2] = np.transpose(W[2])

    
    for i in range(100):
        a_1,z_1 = ut.forward_ae(xe, W[1])
        a_2,z_2 = ut.forward_ae(a_1, W[2])
        A = {0: xe, 1: a_1, 2: a_2}
        Z = {1: z_1, 2: z_2}
        c_n = ut.gradW_ae2(A[2], xe)
        if i%50 == 0:
            print("iteracion numero", str(i), ":",str(c_n))
        W = ut.updW_ae(W, learning_rate, c_n, xe, A, Z) #c_n o E?





    print("Error final: " + str(c_n))
    
    return(W,a_2)         
    return()

#SAE's Training 
def train_sae(xe, p_sae):

    train_ae(xe,p_sae)

#SAE's Training 
# def train_sae(xe,p_sae):
#     W = {}
#     Z = {}
#     A = {}
#     A["a0"] = xe
#     W["w1"] = ut.iniW(int(p_sae[2]),xe.shape[0])
#     W["w2"] = ut.iniW(int(p_sae[2]),xe.shape[0])    
#     for i in range(int(p_sae[0])):
#       Z,A = ut.forward_ae(xe,W,Z,A)
#       mse = ut.gradW_ae(np.transpose(A["a2"]),xe)
#       W = ut.updW_ae(W,A,Z,i,mse,p_sae[1])
#       print("mse :", mse)
#     return(W,A) 
   
   
# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()        
    xe,ye = ut.load_data_csv('data/dtrain.csv')
    W,Xr = train_sae(xe,p_sae)
    Ws, cost  = train_softmax(Xr,ye,p_sft)
    # ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

