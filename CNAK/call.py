import CNAK as Pd
import datetime
import os
import csv
import numpy as np
def readOriginalData(path,fileName,dim,normalization=True):
	os.chdir(path)
	X=np.genfromtxt(fileName, dtype=float, delimiter=",",usecols=range(dim))
	data=X[:,:dim]
	ground_truth=np.genfromtxt(fileName, dtype=float, delimiter=",",usecols=(dim)) #X[:,2]
	#print(np.unique(ground_truth))
	if normalization :
		for i in range(dim):
			data[:,i]/=max(data[:,i])
	#print("data normalization done")
	r, c=data.shape
	tmp=[]
	count=0
	for i in range(r):
		if not (np.sum(data[i,:])==0):
			tmp.append(data[i,:])
		else:
			#print(data[i,:])
			count=count+1
	#print("count:",count)
	data=np.asarray(tmp)
	print(data.shape)
	
	return data, ground_truth


#sample size=7.353294607054592% of original data
gamma=0.27
data=[]
#file1="sim4_"+str(j)+".csv"

path="/home/newdatasetCreation/simulations/sim2"
fileName="sim2_1.csv"
normalization=False
dim=2
k_max=21
data, ground_truth=readOriginalData(path,fileName,dim,normalization)
#print(len(data[0]))
NMI=[]
SIL=[]

#50 realization for a simulation. It computes average metric from 50 execution
for i in range(1,50):
		#path="/home/jayasree/jayasree/2018/first_work/newdatasetCreation/simulations/sim1/"
		os.chdir(path)
		directory=path+"/analysis_T2/CNAK"
		if not os.path.exists(directory):
    			os.makedirs(directory)
		os.chdir(directory)
		directory=str(i+1)
		if not os.path.exists(directory):
    			os.makedirs(directory)
		os.chdir(directory)
		time1= datetime.datetime.now()
		score=[]
		for k in range(1,k_max):

			avg_score, nmi, sil=Pd.main(gamma,data,dim,ground_truth,k,False)

			score.append(avg_score)
			
			file=open("stat.txt","a")
			file.write("\n.............................................................")
			file.close()
			
		K_hat=score.index(min(score))
		print("K_hat:",K_hat+1)
		avg_score, nmi, sil=Pd.main(gamma,data,dim,ground_truth,K_hat+1,True)
		NMI.append(nmi)
		SIL.append(sil)
		time2= datetime.datetime.now()
		val=time2-time1
		
		#print time3-time1
		file=open("stat.txt","a")
		file.write("\nTime:")
		file.write(str(val))
		file.write("\n.............................................................")
		file.close()
		
		#print (time2-time1)
		#print ("time:", time2-time1)

print("NMI--->",NMI)
print("SM-->NMI--->",np.mean(NMI))




print("silhoutte--->",SIL)
print("SM-->silhoutte--->",np.mean(SIL))





