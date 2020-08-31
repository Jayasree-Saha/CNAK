# CNAK
Cluster number assisted K-means
There are two modules
a) sample size: determination of the size of a sample
b) CNAK : Algorithm CNAK

Sample size :  
              i) Provide path, filename and dimension of the dataset in call.py.
              ii)  output gamma
              
CNAK : 
         i) Provide path, filename and dimension of the dataset in call.py. Also provide gamma (output of sample size), k_max
         ii) output K_hat, clustering solution
         
Dataset : 

          i) one simulation of sim-2 is attached 
          
 Note :
 
        In Sample size, we have used hueristic adopted for sim-2. You may consider our paper to understand the value of variable c, tau etc.
  
Dependencies: 
              CNAK has the following dependencies: - numpy - scipy - scikit-learn - munkre

        
 Citation: 
 
 If you find our code useful, please cite the following paper.
 
@article{SAHA2020107625,
title = "CNAK : Cluster Number Assisted K-means",
journal = "Pattern Recognition",
pages = "107625",
year = "2020",
issn = "0031-3203",
doi = "https://doi.org/10.1016/j.patcog.2020.107625",
url = "http://www.sciencedirect.com/science/article/pii/S0031320320304283",
author = "Jayasree Saha and Jayanta Mukherjee",

}

