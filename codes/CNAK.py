#!/usr/bin/env python
#
#~~~~~~~~~~~~~~~~~Cluster Number Assisted K-means (CNAK)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Jayasree Saha, Jayanta Mukherjee, "CNAK: Cluster number assisted K-means",Pattern Recognition, Vol. 110, pp.107625, 2021 
#-----------------Developed by Jayasree Saha 
"""
Introduction
============

The CNAK module provides a K-means based algorithm which does not require pre-declaration of  K. It can detect appropriate
cluster number within the data and useful for handling big data.

This method can solve a few pertinent issues of clustering a dataset: 

1) detection of a single cluster in the absence of any other cluster in a dataset, 
2) the presence of hierarchy,
3) clustering of a high dimensional dataset,
4) robustness over dataset having cluster imbalance, and 
5) robustness to noise.

However, This method is not applicable to different shaped dataset



input(s) : data, gamma(optional), k_max(optional)
output(s) : cnak scores, cluster labels
"""
# -----------------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------------
import os
import random
import munkres
import numpy as np
from sklearn.cluster import KMeans
import argparse

"""
Bucketization of similar centroids.
"""

def Bucketization(cluster1,cluster2,indexes,turn,centers):
	if turn==0:
		for i in range(len(cluster1)):
			centers[i].append(cluster1[i])
	  
	
	for i in range(len(cluster2)):
		r,c=indexes[i]
		centers[r].append(cluster2[c])

	return centers



"""
Matching between set of centroid. This function is called when T_E <= 5
"""
def weightMatrix(clusters,dd_list,k_centers):
	count=0
	avg_score=0
	K=len(clusters[0])
	for i in range(len(clusters)):
		cluster1=clusters[i]
		for j in range(i+1,len(clusters)):
			cluster2=clusters[j]
			count=count+1
			weight=np.zeros([K,K])
			weight_bk=np.zeros([K,K])
			#~~~~~computation of distance between two sets of centroids~~~~~~~
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])
					weight_bk[m][n]=np.linalg.norm(vec1-vec2)
					weight[m][n]=np.linalg.norm(vec1-vec2)
					
			score=0
			#~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)

			#~~~~~Similar centroids are put in same bucket~~~~~~
			if i==0:
				Bucketization(cluster1,cluster2,indexes,j,k_centers)

			#~~~~~~CNAK score~~~~~~~~~
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			avg_score=avg_score+score
			dd_list.append(score)
	
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers


"""
Matching between set of centroid. This function is called when T_E < T_threshold
"""

def weightMatrixUpdated(global_centroids_list,clusters,dd_list,k_centers,avg_score,count):
	
	avg_score=avg_score*count

	K=len(clusters[0])
	
	for i in range(len(global_centroids_list)):
		cluster1=global_centroids_list[i]
		for j in range(len(clusters)):
			cluster2=clusters[j]
			count=count+1
			weight=np.zeros([K,K])
			weight_bk=np.zeros([K,K])
			#~~~~~computation of distance between two sets of centroids~~~~~~~	
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])
					weight_bk[m][n]=np.linalg.norm(vec1-vec2)
					weight[m][n]=np.linalg.norm(vec1-vec2)
					
			score=0
			#~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)
			#~~~~~Similar centroids are put in same bucket~~~~~~
			if i==0:
				Bucketization(cluster1,cluster2,indexes,j,k_centers)
			#~~~~~~CNAK score~~~~~~~~~
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			dd_list.append(score)
			avg_score=avg_score+score
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers


"""
Core function of CNAK
"""

def CNAK_core(data,gamma,K):

	T_S=1
	T_E=5
	
	centroids_list=[]
	for j in range(T_S,T_E):
		#~~~~~random sampling without repitiion~~~~~~~~
		index=random.sample(range(0, len(data)),int(len(data)*gamma))
		samples=[]
		for k in range(int(len(data)*gamma)):
			temp=data[index[k]]
			samples.append(temp)
		#~~~~~~K-means++ on sampled dataset~~~~~~~~~
		kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(samples)
		centroids=kmeans.cluster_centers_
		centroids_list.append(centroids)	
	dd_list=[]
	k_centers=[[] for i in range(len(centroids))]
	#~~~~~~Computation of CNAK score and forming K buckets with T_E similar  centroids~~~~~~~~
	avg_score, count, dd_list,k_centers=weightMatrix(centroids_list,dd_list,k_centers)
	
	#~~~~~Estimate the value of T ~~~~~~~~~
	mean=np.mean(dd_list)
	std=np.std(dd_list)
	val=(1.414*20*std)/(mean)
	
	global_centroids_list=[]
	for centroids in (centroids_list):
		global_centroids_list.append(centroids)
	centers=[[] for i in range(len(centroids_list[0]))]
	
	#~~~~~Repeat untill  T_E > T_threshold ~~~~~~~~~
	while val>T_E:
		T_S=T_E
		T_E=T_E+1
		centroids_list=[]
		for j in range(T_S,T_E):
			index=random.sample(range(0, len(data)),int(len(data)*gamma))
			datax=[]
			
			for k in range(int(len(data)*gamma)):
				temp=data[index[k]]
				datax.append(temp)
			
			kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(datax)
			centroids=kmeans.cluster_centers_
			centroids_list.append(centroids)
	
		avg_score, count, dd_list,k_centers=weightMatrixUpdated(global_centroids_list,centroids_list,dd_list,k_centers,avg_score,count)
		for centroids in ((centroids_list)):
			global_centroids_list.append(centroids)
		mean=np.mean(dd_list)
		std=np.std(dd_list)
		val=(1.414*20*std)/(mean)
		
	clusterCenterAverage=[]
	for i in range(len(k_centers)):
		  clusterCenterAverage.append(np.mean(k_centers[i],axis=0))
	return val, T_E, avg_score, clusterCenterAverage


"""
Generating cluster Label for K_hat
"""

def LabelGeneration(data,k_centers):
	clusterLabel=[]
	
	clusters=[]
	for i in range(len(data)):
		datax=data[i]
		min=np.linalg.norm(np.array(datax)-k_centers[0])
		ClusterIndex=0
		for j in range(1,len(k_centers)):
			datax=data[i]
			temp=np.linalg.norm(np.array(datax)-k_centers[j])
			if temp<min:
			    min=temp
			    ClusterIndex=j
			    
		clusterLabel.append(ClusterIndex)

	
	file=open("CNAK_labels.txt","a")

	for i in range(len(clusterLabel)):
		file.write(str(clusterLabel[i]))
		file.write(str("\n"))
	file.close()
	

	return 

"""
CNAK Implementation
gamma and k_max are optional paraneters. The heuristic used in CNAK paper, can be used for computing gamma. 
"""


def CNAK(data, gamma=0.7, k_min=1, k_max=21):

	#assert data, "data can not be empty!"
		

	CNAK_score=[]
	k_max_centers=[]
	file=open("CNAK_scores.csv","a")
	for K in range(1,k_max):

		val, T_E, avg_score, k_centers=CNAK_core(data,gamma,K)
		CNAK_score.append(avg_score)
		k_max_centers.append(k_centers)
		file.write(str(K))
		file.write(str(","))
		file.write(str(avg_score))
		file.write(str("\n"))
		
	file.close()
	K_hat=CNAK_score.index(min(CNAK_score))
	print("K_hat:",K_hat+1)
	#~~~~~~~~Labels for K_hat~~~~~~~~~
	LabelGeneration(data,k_max_centers[K_hat])

	return




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-p","--path", type=str, help="Provide location of input file. If it is empty, file should be in the current directory")
	parser.add_argument("-f","--file", type=str, help="Provide name of inpute file.")
	parser.add_argument("-k1","--K_min", type=int, help="It is optional. It denotes the lower limit of cluster number that you assume for the dataset. Default value 1")
	parser.add_argument("-k2","--K_max", type=int, help="It is optional. It denotes the upper limit of cluster number that you assume for the dataset. Default value 21")
	parser.add_argument("-g","--gamma", type=float, help="It is optional. It denotes fraction of sample. Value can be 0-1. Default value 0.7")
	args = parser.parse_args()
	assert args.file, "File name can not be empty" 
	print("CNAK....")
	
	if not args.path:
		fileName=args.file
	else :
		fileName=args.path.rstrip('/')+"/"+args.file
	
	##load data
	X=np.genfromtxt(fileName, dtype=float, delimiter=",")
	
	gamma=args.gamma
	k_min=args.K_min
	k_max=args.K_max
	print(" gamma:",gamma," K_min:",k_min," K_max:",k_max)
	#CNAK
	CNAK(X,  gamma, k_min, k_max)
