import librosa
import librosa.display
import IPython.display as ipd
import sklearn
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import operator
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import sys
import scipy
import warnings
warnings.simplefilter("ignore")


def choose_best_cluster_kmeans(data):
    scaled_pca_df,scal=scale_PCA_data(data)    
    clusters=[]     
    se_difference=[]
    se_difference.append(0)
    se={}
    kFold = KFold(n_splits=5, shuffle=True, random_state=32)
    for train_index,test_index in kFold.split(scaled_pca_df):
        train_data = scaled_pca_df.iloc[train_index]
        test_data = scaled_pca_df.iloc[test_index]
        for k in range(1,10):            
            k_m = KMeans(n_clusters=k, max_iter=1000).fit(train_data.iloc[:,:-1])           
            se[k]=k_m.inertia_
            if k>1:
                se_difference.append((se[k-1] -se[k])/se[1]) 
        best_cluster_num=len([x for x in se_difference if x>0.05])+1
        clusters.append(best_cluster_num)    
    best_cluster=Counter(clusters).most_common(1)[0][0]
    return best_cluster


def validation_kmeans(data,best_cluster):    
    scaled_pca_df,scal=scale_PCA_data(data)    
    scaled_pca_df["predicted_label"]=-1
    re_cluster=True
    accuracies={}
    accuracy_labels_dicts={}
    k_ms={}
    kFold = KFold(n_splits=5, shuffle=True, random_state=None)
    for i,(train_ind,test_ind) in enumerate(kFold.split(scaled_pca_df)):
        train_data = scaled_pca_df.iloc[train_ind]
        test_data = scaled_pca_df.iloc[test_ind]
        for tr in range(40):
            train_data["predicted_label"]=-1
            k_m=KMeans(n_clusters=best_cluster, max_iter=1000).fit(train_data.iloc[:,:-2])
            lables_list=[]
            train_data["predicted_cluster"]=k_m.labels_
            data,labels_dict=assign_most_freq_label(train_data,range(best_cluster))            
            if  ("BG" in list(set(labels_dict.values())))&(len(set(labels_dict.values()))==best_cluster):
                test_data,accuracy=evaluate_test(test_data,k_m,best_cluster,labels_dict)
                accuracies[i]=accuracy
                accuracy_labels_dicts[i]=labels_dict
                break
            else:
                train_data=train_data.drop(columns=["predicted_cluster"]) 
    item_best_accuracy=max(accuracies.items(), key=lambda x : x[1])
    best_accuracy_key=item_best_accuracy[0]
    best_accuracy=item_best_accuracy[1]
    best_dict=accuracy_labels_dicts[best_accuracy_key]
    performance=np.mean(list(accuracies.values()))
    return k_m,performance,best_dict,scal
                                      
def assign_most_freq_label(data,best_cluster):
    labels_dict={}        
    for k in best_cluster:
        part_data=data[data["predicted_cluster"]==k]
        labels_part=part_data["labels"].values
        if (len(Counter(labels_part).most_common(1))!=0):
            most_freq_label=Counter(labels_part).most_common(1)[0][0]
            data.loc[data[data["predicted_cluster"]==k].index,"predicted_label"]=most_freq_label
            labels_dict[k]=most_freq_label
    return data,labels_dict       
                                      

def scale_PCA_data(data):
    #Scale
    x_data=data.iloc[:,1:-1]
    y_data=data.iloc[:,-1:]
    scaler=StandardScaler()
    scaled_x = scaler.fit_transform(np.array(x_data, dtype = float))    
    #Take PCA
    pca=PCA(n_components = 0.85)
    scaled_pca_x=pca.fit_transform(scaled_x)
    scaled_pca_dataf=pd.DataFrame(scaled_pca_x)
    scaled_pca_dataf["labels"]=y_data
    return scaled_pca_dataf,scaler

def evaluate_test(data,kmeans,b_cluster,labels_dict):
    predicted_labels=kmeans.predict(data.iloc[:,:-2])
    data["predicted_cluster"]=predicted_labels
    data,_=assign_most_freq_label(data,range(b_cluster))
    
    data=data.rename(columns={"predicted_label":"actual_label"})    
    data["predicted_label"]=data["predicted_cluster"]
    data["predicted_label"]=data["predicted_label"].replace(labels_dict) 
    diff_labels=set(data["predicted_label"]).symmetric_difference(set(data["actual_label"]))
    
    rep_dict={k:"BG" for k in diff_labels}
    data["predicted_label"]=data["predicted_label"].replace(rep_dict) 
    data["actual_label"]=data["actual_label"].replace(rep_dict) 
    acc=accuracy_score(data["predicted_label"],data["actual_label"])
    return data,acc