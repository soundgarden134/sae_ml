# Deep-Learning:Training via BP+GD
import pandas     as pd
import numpy      as np
import my_utility as ut
import math
	
# Softmax's training
def train_softmax(Xr,ye,p_sft):  
    num_nodes = Xr.shape[0]
    
    num_iter = int(p_sft.iloc[0].values[0])
    learning_rate = int(p_sft.iloc[1].values[0])
    penalty_rate = int(p_sft.iloc[2].values[0])
    num_samples = ye.shape[0]
    Ws = {1: ut.iniW(Xr.shape[0], ye.shape[1])}
    
    for i in range(num_iter):
        
        z = np.dot(np.transpose(Xr),Ws[1])    
        a = ut.softmax(z)
        a = ut.forward_softmax(Xr, Ws[1])
        cost, gW = ut.grad_softmax(a,ye, Ws,Xr, penalty_rate, learning_rate)
        Ws[1] = Ws[1] - learning_rate*gW

        

    
    
    return Ws,cost

# AE's Training 
def train_ae(input_nodes, num_nodes, num_iter, learning_rate):

    
    n0 = input_nodes.shape[0]
    W = {}
    W = {1: ut.iniW(num_nodes, n0), 2: ut.iniW(num_nodes,n0)}
    W[2] = np.transpose(W[2])

    
    for i in range(num_iter):
        a_1,z_1 = ut.forward_ae(input_nodes, W[1])
        a_2,z_2 = ut.forward_ae(a_1, W[2])
        A = {0: input_nodes, 1: a_1, 2: a_2}
        Z = {1: z_1, 2: z_2}
        E, c_n = ut.gradW_ae(A[2], input_nodes)
        W = ut.updW_ae(W, learning_rate, E, input_nodes, A, Z) #c_n o E?


    
    
    
    return W[1], A   


#SAE's Training 
def train_sae(xe, p_sae):
    n0 = int(xe.shape[0])
    n1 = int(p_sae.iloc[2].values[0])
    n2 = int(p_sae.iloc[3].values[0])
    num_iter = int(p_sae.iloc[0].values[0])
    learning_rate = p_sae.iloc[1].values[0]
    W = {}
    W[1], A = train_ae(xe, n1, num_iter, learning_rate)
    W[2], A = train_ae(A[1], n2, num_iter, learning_rate)
    
    
    return W, A[1]


   
# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()        
    xe,ye = ut.load_data_csv('data/dtrain.csv')
    W,Xr = train_sae(xe,p_sae)
    Ws, cost  = train_softmax(Xr,ye,p_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

