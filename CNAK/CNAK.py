import os
import struct
import csv
import numpy as np
import random
import datetime
import munkres
#from scipy.cluster.vq import kmeans2 as cnak_plus
#import scipy.cluster.vq as km
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cluster as nmi_score
def hashClusterCenters(cluster1,cluster2,indexes,turn,centers):
	if turn==0:
		for i in range(len(cluster1)):
			centers[i].append(cluster1[i])
	  
	
	for i in range(len(cluster2)):
		r,c=indexes[i]
		centers[r].append(cluster2[c])

	return centers


#hungarian method
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
			weight_bk=np.zeros(shape=(len(cluster1),len(cluster2)))
				
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])

					weight[m][n]=np.linalg.norm(vec1-vec2)
					weight_bk[m][n]=np.linalg.norm(vec1-vec2)
			score=0
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)
			if i==0:
				hashClusterCenters(cluster1,cluster2,indexes,j,k_centers)
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			dd_list.append(score)
			avg_score=avg_score+score
	
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers



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
				
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])

					weight[m][n]=np.linalg.norm(vec1-vec2)
					weight_bk[m][n]=np.linalg.norm(vec1-vec2)
			score=0
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)
			if i==0:
				hashClusterCenters(cluster1,cluster2,indexes,j,k_centers)
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			dd_list.append(score)
			avg_score=avg_score+score
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers



def dataCreate(data,proportion,dim,K):

	T_S=1
	T_E=5
	
	centroids_list=[]
	for j in range(T_S,T_E):#100
		index=random.sample(range(0, len(data)),int(len(data)*proportion))
		datax=[]
		for k in range(int(len(data)*proportion)):
			temp=data[index[k]]
			datax.append(temp)

		kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(datax)
		clusterLabel=kmeans.labels_
		centroids=kmeans.cluster_centers_
		centroids_list.append(centroids)	
	dd_list=[]
	k_centers=[[] for i in range(len(centroids))]
	avg_score, count, dd_list,k_centers=weightMatrix(centroids_list,dd_list,k_centers)
	
	mean=np.mean(dd_list)
        std=np.std(dd_list)
        val=(1.414*20*std)/(mean)
	#print("centroids_list shape:",np.asarray(centroids_list).shape)
	global_centroids_list=[]
	for centroids in (centroids_list):
		#print("centroids shape:",np.asarray(centroids).shape)
		global_centroids_list.append(centroids)
	centers=[[] for i in range(len(centroids_list[0]))]
	#print("before update T_lemma", val)
	
	while val>T_E:
		#print("val>T_E")
		T_S=T_E
		T_E=T_E+1
		centroids_list=[]
		for j in range(T_S,T_E):#100
			index=random.sample(range(0, len(data)),int(len(data)*proportion))
			datax=[]
			
			for k in range(int(len(data)*proportion)):
				temp=data[index[k]]
				datax.append(temp)
			#print("fraction:",len(datax))
			kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(datax)
			clusterLabel=kmeans.labels_
			centroids=kmeans.cluster_centers_
			centroids_list.append(centroids)	
		avg_score, count, dd_list,k_centers=weightMatrixUpdated(global_centroids_list,centroids_list,dd_list,k_centers,avg_score,count)
		for centroids in ((centroids_list)):
			global_centroids_list.append(centroids)
		mean=np.mean(dd_list)
        	std=np.std(dd_list)
        	val=(1.414*20*std)/(mean)
		#print("after update T_lemma", val)
		#print("inside update T", T_E)
	#print("data_creation_done")
	
	os.chdir("..")
	return val, T_E, avg_score, k_centers


def Bucketization(data,k_centers,ground_truth):
	clusterLabel=[]
	dataLabel=[]
	clusterCenterAverage=[[] for i in range(len(k_centers))]
	#print("bucketization::K=",len(k_centers))
	for i in range(len(k_centers)):
		  clusterCenterAverage[i].append(np.mean(k_centers[i],axis=0))
	clusters=[]
	for i in range(len(data)):
		datax=data[i]
		#index=dim
		min=np.linalg.norm(np.array(datax)-clusterCenterAverage[0])
		ClusterIndex=0
		for j in range(1,len(clusterCenterAverage)):
			datax=data[i]
			temp=np.linalg.norm(np.array(datax)-clusterCenterAverage[j])
			if temp<min:
			    min=temp
			    ClusterIndex=j
			    
		clusterLabel.append(ClusterIndex)

	#print("final cluster labels:",clusterLabel)
	
	label= "clustering metrics:\n"
	file=open("stat.txt","a")
	file.write(label)
	file.close()
	val=nmi_score.normalized_mutual_info_score(ground_truth,clusterLabel)
	file=open("stat.txt","a")
	file.write("\nNMI:")
	file.write(str(val))
	file.close()
	print("NMI:",val)
	
	val=metrics.silhouette_score(data, clusterLabel, metric='euclidean')
	file=open("stat.txt","a")
	file.write("\nsil:")
	file.write(str(val))
	file.close()
	
	nmi=0
	sil=0
	
	nmi=nmi_score.normalized_mutual_info_score(ground_truth,clusterLabel)
	sil=metrics.silhouette_score(data, clusterLabel, metric='euclidean')

	return nmi, sil

def main(proportion,data,dim,ground_truth,K,Testing):#proportion
	directory='sample_'+str(proportion*100)
	if not os.path.exists(directory):
    		os.makedirs(directory)
	os.chdir(directory)	
	val, T_E, avg_score, k_centers=dataCreate(data,proportion,dim,K)

	nmi=0
	sil=0

	if Testing:
		
		nmi, sil =Bucketization(data,k_centers,ground_truth)
	return avg_score, nmi, sil
	
#proportion
	
if __name__=="main":
    main()
