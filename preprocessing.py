import pandas as pd
import numpy as np
import my_utility as ut
 


# Beginning ...
def main():			
	xv,yv  = ut.load_data_csv('dtest.csv')
	W      = ut.load_w_dl()
	zv     = ut.forward_dl(xv,W)      		
	ut.metricas(yv,zv) 	
	
	

if __name__ == '__main__':   
	 main()

