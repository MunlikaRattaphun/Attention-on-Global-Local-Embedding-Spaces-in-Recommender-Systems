# -*- coding: utf-8 -*-

from scipy.spatial import distance
from util import pool_candidate
import pandas as pd
import numpy as np

def evaluation_model(model_name, model, num_users, cluster_predict, num_retrieve2, cluster_member, dataset, gt, GD_users, GD_items, GP_users, GP_items, LD_users, LD_items, LP_users, LP_items, topk):
    
    sum_recall = 0
    sum_precision = 0
    sum_hr = 0
    sum_arhr =0     
    sum_candidate = 0
    count = 0   
     
    # 1. generate candidate set for each user based on number of retrieve clusters
    candidate_items = pool_candidate(num_users, cluster_predict, num_retrieve2, cluster_member)

    for i in range(num_users):

        GP_user_test = []
        GP_pitem_test = []
        GD_user_test = []
        GD_pitem_test = []
        LP_user_test = []
        LP_pitem_test = []
        LD_user_test = []
        LD_pitem_test = []
        ui_list = []

        # 1. get all positive item 
        filter_user = dataset.loc[dataset['u_id'] == i]
        all_pitems = filter_user['i_id']
        all_pitems = all_pitems.tolist() 

        # 2. get gt item 
        gt_set = gt[i]

        # 3. get candidate item
        candidate = candidate_items[i]
        
        # delete train positive from candidate set
        train_positive = set(all_pitems) - set(gt_set)
        test_candidate = list(set(candidate) - train_positive)
        sum_candidate += len(test_candidate)            
        
        if len(test_candidate) ==0:
            recall = 0
            precision = 0
            hr = 0
            arhr = 0
            count +=1
            continue
        else:            
            for j in test_candidate:
                GP_user_test.append(GP_users[i])
                GP_pitem_test.append(GP_items[j])
                GD_user_test.append(GD_users[i])
                GD_pitem_test.append(GD_items[j])
                LP_user_test.append(LP_users[i])
                LP_pitem_test.append(LP_items[j])
                LD_user_test.append(LD_users[i])
                LD_pitem_test.append(LD_items[j])
                ui_list.append([i,j])
    
            GP_user_test = np.asarray(GP_user_test)    
            GP_pitem_test = np.asarray(GP_pitem_test)          
            GD_user_test = np.asarray(GD_user_test)    
            GD_pitem_test = np.asarray(GD_pitem_test)
            LP_user_test = np.asarray(LD_user_test)
            LP_pitem_test = np.asarray(LD_pitem_test)
            LD_user_test = np.asarray(LP_user_test)
            LD_pitem_test = np.asarray(LP_pitem_test)
            
            user_blend, item_blend, item_blend2 = model.predict([GD_user_test, GP_user_test, LD_user_test, LP_user_test, GD_pitem_test, GP_pitem_test, LD_pitem_test, LD_pitem_test, GD_pitem_test, GP_pitem_test, LD_pitem_test, LP_pitem_test])
            if model_name == "AD":
                recall, precision, hr, arhr  = eva_euclidean(user_blend, item_blend, ui_list, gt_set, topk)
                sum_recall += recall
                sum_precision += precision
                sum_hr += hr
                sum_arhr += arhr
            else:
                recall, precision, hr, arhr  = eva_dot_product(user_blend, item_blend, ui_list, gt_set, topk)
                sum_recall += recall
                sum_precision += precision
                sum_hr += hr
                sum_arhr += arhr              
                
    num_users = num_users-count
    print ("%s\t %d \t\t %0.4f \t\t %0.4f \t\t %0.4f \t\t %0.4f \t\t%0.2f" %(model_name, topk, (sum_recall/num_users), (sum_precision/num_users), (sum_hr/num_users), (sum_arhr/num_users), (sum_candidate/num_users)))


def eva_dot_product(user_vector, item_vector, ui_list, gt_set, Topk):
    
    recall = 0
    precision = 0
    HR = 0
    ARHR = 0
    dist_table = []
    
    for i in range(user_vector.shape[0]):
        itm_dist = np.dot(user_vector[i], item_vector[i].transpose())
        dist_table.append([ui_list[i][0], ui_list[i][1], itm_dist])
        
    df = pd.DataFrame(dist_table)
    df.columns = ['u_id', 'i_id', 'dist']
    sorted_df = df.sort_values(by='dist', ascending=False)
    top_k_items = sorted_df[:Topk]
    top_k_items = top_k_items['i_id']
    top_k_items = top_k_items.tolist()       
    
    # recall precision and HR calculation
    count = len([w for w in gt_set if w in top_k_items])
    if count>0:
        HR+=1
        recall = count/len(gt_set)
        precision = count/Topk
    else: 
        recall = 0
        precision = 0
    
    # ARHR calculation
    for pos in range(len(top_k_items)):
        result = set(list([top_k_items[pos]])) & set(gt_set)
        if len(result) == 1:
            ARHR += (1/(pos+1)) 

    return recall, precision, HR, ARHR


def eva_euclidean(user_vector, item_vector, ui_list, gt_set, Topk):
    
    recall = 0
    precision = 0
    HR = 0
    ARHR = 0
    dist_table = []
    
    for i in range(user_vector.shape[0]):
        itm_dist = distance.euclidean(user_vector[i], item_vector[i]) 
        dist_table.append([ui_list[i][0], ui_list[i][1], itm_dist])
        
    df = pd.DataFrame(dist_table)
    df.columns = ['u_id', 'i_id', 'dist']
    sorted_df = df.sort_values(by='dist', ascending=True)
    top_k_items = sorted_df[:Topk]
    top_k_items = top_k_items['i_id']
    top_k_items = top_k_items.tolist()       
    
    # recall precision and HR calculation
    count = len([w for w in gt_set if w in top_k_items])
    if count>0:
        HR+=1
        recall = count/len(gt_set)
        precision = count/Topk
    else: 
        recall = 0
        precision = 0
    
    # ARHR calculation
    for pos in range(len(top_k_items)):
        result = set(list([top_k_items[pos]])) & set(gt_set)
        if len(result) == 1:
            ARHR += (1/(pos+1)) 

    return recall, precision, HR, ARHR

    
def distance_calculation(predict_user_vec, test_u_i_list, predict_item_post):
    predit_dist = []    
    for i in range(predict_user_vec.shape[0]):
        uid = test_u_i_list[i][0]
        iid = test_u_i_list[i][1]
        itm_dist = distance.euclidean(predict_user_vec[i], predict_item_post[i]) 
        predit_dist.append([uid, iid, itm_dist])

    df = pd.DataFrame(predit_dist)
    df.columns = ['u_id', 'i_id', 'dist']
    return(df)


def evaluate_base_emb(vector_name, user_vector, item_vector, dataset, cluster_predict, retrieve, cluster_member, gt, topK):
    num_users = len(user_vector)
    sum_recall =0
    sum_precision = 0
    sum_hr = 0
    sum_arhr = 0
    sum_candidate = 0
        
    candidate_items = pool_candidate(len(user_vector), cluster_predict, retrieve, cluster_member)  
    for i in range(len(user_vector)):
        
        user_test = []
        pitem_test = []
        ui_list = []

        # get all positive item 
        filter_user = dataset.loc[dataset['u_id'] == i]
        all_pitems = filter_user['i_id']
        all_pitems = all_pitems.tolist() 

        # get gt item 
        gt_set = gt[i]

        # get candidate item
        candidate = candidate_items[i]
        
        train_positive = set(all_pitems) - set(gt_set)
        test_candidate = list(set(candidate) - train_positive)
        sum_candidate += len(test_candidate)
        if len(test_candidate) ==0:
            recall = 0
            precision = 0
            hr = 0
            arhr = 0
        else:
            
            for j in test_candidate:
                user_test.append(user_vector[i])
                pitem_test.append(item_vector[j])
                ui_list.append([i,j])
    
            user_test = np.asarray(user_test)    
            pitem_test = np.asarray(pitem_test)
            recall, precision, hr, arhr  = eva_euclidean(user_test, pitem_test, ui_list, gt_set, topK)            
            
        sum_recall += recall
        sum_precision += precision
        sum_hr += hr
        sum_arhr += arhr

    print ("%s\t %d \t\t %0.4f \t\t %0.4f \t\t %0.4f \t\t %0.4f \t\t%0.2f" %(vector_name, topK, (sum_recall/num_users), (sum_precision/num_users), (sum_hr/num_users), (sum_arhr/num_users), (sum_candidate/num_users)))


