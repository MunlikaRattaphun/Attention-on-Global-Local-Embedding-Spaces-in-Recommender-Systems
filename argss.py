# -*- coding: utf-8 -*-
"""
@author: Kero
"""

class parser(object):
    def __init__(self):
        dataset_name = "yelp_mcity"
        self.path = 'input'
        self.f_GD_user = 'GD_'+str(dataset_name)+'_user_emb.txt'
        self.f_GD_item = 'GD_'+str(dataset_name)+'_item_emb.txt'
        self.f_GP_user = 'GP_'+str(dataset_name)+'_user_emb.txt'
        self.f_GP_item = 'GP_'+str(dataset_name)+'_item_emb.txt'
        self.f_LD_user = 'LD_'+str(dataset_name)+'_user_emb_M30.txt'
        self.f_LD_item = 'LD_'+str(dataset_name)+'_item_emb_M30.txt'
        self.f_LP_user = 'LP_'+str(dataset_name)+'_user_emb_M30.txt'
        self.f_LP_item = 'LP_'+str(dataset_name)+'_item_emb_M30.txt'      
        self.f_dataset = str(dataset_name)+'_interactions.txt'
        self.f_gt = str(dataset_name)+'_gt.txt'
        self.f_train_uid = str(dataset_name)+'_train_uid.txt'
        self.f_train_pid = str(dataset_name)+'_train_pid.txt'
        self.f_train_nid = str(dataset_name)+'_train_nid.txt'        
        self.f_idx_cluster = str(dataset_name)+'_index_user_cluster_M30.txt'
        self.f_cluster_member = str(dataset_name)+'_cluster_member_M30.txt'
        self.model_name = "AD"
        self.num_epochs = 100
        self.num_batch = 128
        self.emb_dim = 64
        self.K = [2] 
        self.topk = 20
