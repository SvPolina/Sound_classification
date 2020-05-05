import librosa
import librosa.display
import IPython.display as ipd
import sklearn
import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import operator
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import sys
import scipy
import copy
from KMeans_F import scale_PCA_data, assign_most_freq_label
import warnings
warnings.simplefilter("ignore")



#KNN

def choose_best_knn(data):    
    scaled_pca_df,scal=scale_PCA_data(data) 
    clusters=[]  
    se={}
    kFold = KFold(n_splits=5, shuffle=True, random_state=22)
    accuracy_dicts={}    
    for i,(train_ind,test_ind) in enumerate(kFold.split(scaled_pca_df)):        
        train_data = scaled_pca_df.iloc[train_ind]
        train_data=train_data.reset_index(drop=True)
        test_data = scaled_pca_df.iloc[test_ind]
        test_data=test_data.reset_index(drop=True)
        accuracy_dict={}
        for k in range(8, 23):
            accuracy_dict[k],num_of_clus=caluculate_accuracy(k,train_data,test_data,test_mode=False)
            if num_of_clus<4:
                break
        accuracy_dicts[i]= accuracy_dict 
    accuracies_df=pd.DataFrame(accuracy_dicts)
    accuracies_df["avg_acc"]=accuracies_df.mean(axis=1)
    best_nn=accuracies_df[accuracies_df["avg_acc"]==np.max(accuracies_df["avg_acc"].values)].index[0]
    best_acc=accuracies_df[accuracies_df["avg_acc"]==np.max(accuracies_df["avg_acc"].values)]["avg_acc"].values[0]
    return best_nn,best_acc

def cluster_gen():
    n=1
    for i in range(1000):
       n=i
       yield n        
      
def assign_cluster(data,neigh_array):
    c_gen=cluster_gen()
    for i,row in enumerate(neigh_array):
        indexes_in_c=np.append(np.array(data.index[i]), row)
        cur_labels=data.iloc[indexes_in_c]['predicted_cluster'].values        
        fev_label = Counter(cur_labels).most_common(1)[0][0]
        if fev_label==-1:
            data.loc[data.iloc[np.array(indexes_in_c)].index,'predicted_cluster']=next(c_gen)
        else:
            data.loc[data.iloc[np.array(indexes_in_c)].index,'predicted_cluster']=fev_label
    return data 

def test_KNN(best_nn,train_data,test_data):
    scaled_pca_train,scal=scale_PCA_data(train_data)
    x_data_test=test_data.iloc[:,1:-1]
    y_data_test=test_data.iloc[:,-1:]
    scaled_x_test=scal.transform(x_data_test)

    pca=PCA(n_components = 0.85)
    scaled_pca_x=pca.fit_transform(scaled_x_test)
    scaled_pca_test=pd.DataFrame(scaled_pca_x)
    scaled_pca_test["labels"]=y_data_test.values
    test_accuracy,_=caluculate_accuracy(best_nn,scaled_pca_train,scaled_pca_test,test_mode=True)    
    return test_accuracy

def caluculate_accuracy(nn_num,tr_data_c,tst_data_c,test_mode=False):
    tr_data=copy.deepcopy(tr_data_c)
    tst_data=copy.deepcopy(tst_data_c)
    neigh  = NearestNeighbors(nn_num)
    k_neigh = neigh.fit(tr_data.iloc[:,:-1])
    nn_neighb=k_neigh.kneighbors(tr_data.iloc[:,:-1], return_distance=False) 
    tr_data["predicted_cluster"]=-1
    tr_data["predicted_cluster_dup"]=-1
    tr_data=assign_cluster(tr_data,nn_neighb) 
    tr_data,labels_dict=assign_most_freq_label(tr_data,list(set(tr_data["predicted_cluster"])))            
    tr_data["predicted_cluster_dup"]=tr_data["predicted_cluster"]            
    tst_data["predicted_cluster"]=-1
    tst_data["predicted_cluster_dup"]=-1
    tst_data["predicted_label"]=-1
            
    total_data=pd.concat((tr_data,tst_data))
    total_data=total_data.reset_index(drop=True)            
    nn_neighb=k_neigh.kneighbors(total_data.iloc[:,:-4], return_distance=False)
    total_data=assign_cluster(total_data,nn_neighb)
    old_test=total_data[total_data["predicted_cluster_dup"]==-1]
    old_train=total_data[total_data["predicted_cluster_dup"]!=-1]
    
    old_train=old_train.rename(columns={"predicted_label":"actual_label"})
    old_test,_=assign_most_freq_label(old_test,list(set(old_test["predicted_cluster"])))

    old_test=old_test.rename(columns={"predicted_label":"actual_label"})
    old_test["predicted_label"]=old_test["predicted_cluster"]
    old_test["predicted_label"]=old_test["predicted_label"].replace(labels_dict)
    accuracy=accuracy_score(old_test["predicted_label"],old_test["actual_label"])
    num_of_clust=len(set(old_test["predicted_label"]))
    if test_mode:
        print("The number of clusters is",num_of_clust)
    tr_data=tr_data.drop(columns=["predicted_cluster","predicted_cluster_dup","predicted_label"])  
    tst_data=tst_data.drop(columns=["predicted_cluster","predicted_cluster_dup","predicted_label"])
    return accuracy,num_of_clust