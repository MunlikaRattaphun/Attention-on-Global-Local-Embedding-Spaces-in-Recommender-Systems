# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:50:47 2021

@author: Kero
"""
import numpy as np
import pandas as pd

def interaction_matrix(fname):
    
    dataset = pd.read_csv(fname, sep="\t", usecols= [0,1,2], header=None)
    dataset.columns = ['u_id', 'i_id', 'rating']
    
    return dataset

def load_cmember(fname):
    
    cluster_member = []    
    with open(fname, "r") as cidxf:
        for cluster in cidxf:
            Line = cluster.replace(" \n",'')
            cluster_member.append([ int(x) for x in Line.split(" ")])
        del(Line) 
        
    return (cluster_member)

def get_gt_items(fname):
    
    gtset = []
    with open(fname, "r") as cidxf:
        for cluster in cidxf:
            Line = cluster.replace(" \n",'')
            gtset.append([ int(x) for x in Line.split(",")])
        del(Line)   
        
    return(gtset)

def pool_candidate(num_user, idx_cluster, max_index_retrieve, cluster_member):

    user_pool_canditate = []    
    for i in range(num_user):
        
        pool_candidate = set()
        for j in range(max_index_retrieve):
            index_cluster = j
            retrieve_cluster = idx_cluster[:,index_cluster]
            all_item_candidate = get_candidate_item(retrieve_cluster, cluster_member)
            item_candidate = all_item_candidate[i]
            pool_candidate = set(pool_candidate) | set(item_candidate) 
        user_pool_canditate.append(list(pool_candidate))

    return (user_pool_canditate)

def get_candidate_item(idx_cluster, cluster_member):
    
    item_candidate = []  
    for i in range(len(idx_cluster)):
        x = idx_cluster[i]
        candidate = cluster_member[x]   
        item_candidate.append(candidate)
        
    return(item_candidate)

def get_base_emb(train_uid, train_pid, train_nid, user_vector, item_vector):
    
    global_user_vector = []
    global_pitem_vector = []
    global_nitem_vector = []

    for i in range(len(train_uid)):
        uid = train_uid[i]
        pid = train_pid[i]
        nid = train_nid[i]
        global_user_vector.append(user_vector[uid])
        global_pitem_vector.append(item_vector[pid])
        global_nitem_vector.append(item_vector[nid])

    global_user_vector = np.asarray(global_user_vector)    
    global_pitem_vector = np.asarray(global_pitem_vector)          
    global_nitem_vector = np.asarray(global_nitem_vector)   
           
    return(global_user_vector, global_pitem_vector, global_nitem_vector)
