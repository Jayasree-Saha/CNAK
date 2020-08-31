'''
Documentation, License etc.

@package Sample_Size_Determination
'''
import os
import numpy as np
import datetime


def main(path, fileName, dim):
    
    
    os.chdir(path)
    X=np.genfromtxt(fileName, dtype=float, delimiter=",",usecols=range(dim)) 
    Y=X
    X=X.T
    
    cov1=np.cov(X,rowvar=True)
    
    
    
    eigvals, eigvecs = np.linalg.eig(cov1)
    eigvals=sorted(eigvals,reverse=True)
    print ("eigvals:",eigvals)
    
    tau=16
    print("tau:",tau)
    print("lambda_max:",eigvals[0])
    
    c=np.float_power(eigvals[0],(float(1)/tau))
    print ("c=",c)
    sample1=((eigvals[0])*np.float_power((1.96/c),2))
    
    gamma=sample1/(1+sample1/len(Y))
    
        
    print("gamma:%f"%gamma)
    
    print("sample size="+str(gamma*100/len(Y))+"% of original data")
    


    
if __name__=="main":
	main()    
