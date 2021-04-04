# -*- coding: utf-8 -*-
import os
import numpy as np
from argss import parser
from model import ATT_AD, ATT_AP
from evaluate import evaluation_model, evaluate_base_emb
from util import (load_cmember, get_gt_items, interaction_matrix, get_base_emb)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
opts = parser()

dataset_path = os.path.join(opts.path,'dataset', opts.f_dataset)
gt_set_path = os.path.join(opts.path,'dataset', opts.f_gt)
GD_user_path = os.path.join(opts.path,'embedding', opts.f_GD_user)
GD_item_path = os.path.join(opts.path,'embedding', opts.f_GD_item)
GP_user_path = os.path.join(opts.path,'embedding', opts.f_GP_user)
GP_item_path = os.path.join(opts.path,'embedding', opts.f_GP_item)
LD_user_path = os.path.join(opts.path,'embedding', opts.f_LD_user)
LD_item_path = os.path.join(opts.path,'embedding', opts.f_LD_item)
LP_user_path = os.path.join(opts.path,'embedding', opts.f_LP_user)
LP_item_path = os.path.join(opts.path,'embedding', opts.f_LP_item)
idx_cluster_path = os.path.join(opts.path,'cluster', opts.f_idx_cluster)
cmember_path = os.path.join(opts.path,'cluster', opts.f_cluster_member)
train_u_path = os.path.join(opts.path,'dataset', opts.f_train_uid)
train_p_path = os.path.join(opts.path,'dataset', opts.f_train_pid)
train_n_path = os.path.join(opts.path,'dataset', opts.f_train_nid)

##### dataset information and ground truth
dataset =  interaction_matrix(dataset_path)
gt = get_gt_items(gt_set_path)

##### base embedding 
GD_user = np.loadtxt(GD_user_path)
GD_item = np.loadtxt(GD_item_path)
GP_user = np.loadtxt(GP_user_path)
GP_item = np.loadtxt(GP_item_path)
LD_user = np.loadtxt(LD_user_path)
LD_item = np.loadtxt(LD_item_path)
LP_user = np.loadtxt(LP_user_path)
LP_item = np.loadtxt(LP_item_path)
num_users = len(GD_user)
num_items = len(GD_item)

##### top-K clusters
idx_cluster = np.loadtxt(idx_cluster_path,dtype = int)

##### item clusters
cluster_member = load_cmember(cmember_path)

##### training set
train_uid = np.loadtxt(train_u_path, dtype=int)
train_pid = np.loadtxt(train_p_path, dtype=int)
train_nid = np.loadtxt(train_p_path, dtype=int)

GD_user_train, GD_pitem_train, GD_nitem_train = get_base_emb(train_uid, train_pid, train_nid, GD_user, GD_item)
GP_user_train, GP_pitem_train, GP_nitem_train = get_base_emb(train_uid, train_pid, train_nid, GP_user, GP_item)
LD_user_train, LD_pitem_train, LD_nitem_train = get_base_emb(train_uid, train_pid, train_nid, LD_user, LD_item)
LP_user_train, LP_pitem_train, LP_nitem_train = get_base_emb(train_uid, train_pid, train_nid, LP_user, LP_item)

model_name = opts.model_name
if model_name == "AD":
    model = ATT_AD(GD_user_train, GP_user_train, LD_user_train, LP_user_train, GD_pitem_train, GP_pitem_train, LD_pitem_train, LP_pitem_train, GD_nitem_train, GP_nitem_train, LD_nitem_train, LP_nitem_train, opts.emb_dim)   
else:
    model = ATT_AP(GD_user_train, GP_user_train, LD_user_train, LP_user_train, GD_pitem_train, GP_pitem_train, LD_pitem_train, LP_pitem_train, GD_nitem_train, GP_nitem_train, LD_nitem_train, LP_nitem_train, opts.emb_dim)   
  
print ("############################################################################")     
print("training attention user model") 
for epoch in range(opts.num_epochs):
        print('Epoch %s/%s' % (epoch+1, opts.num_epochs))   
        loss = model.fit(x=[GD_user_train, GP_user_train, LD_user_train, LP_user_train, GD_pitem_train, GP_pitem_train, LD_pitem_train, LP_pitem_train, GD_nitem_train, GP_nitem_train, LP_nitem_train, LP_nitem_train],
                  batch_size = opts.num_batch,
                  verbose=0,
                  epochs=1)
        print('train loss:', loss.history['loss'])    
print ("training finished")

for i in opts.K:
    print ("#############################################################################")  
    print ("M = 30, K =", i)
    print("Emb \t top-k \t Recall \t\t Prec \t\t HR  \t\t ARHR \t\t #Candidate.")
    print ("#############################################################################")    
    evaluation_model(model_name, model, num_users, idx_cluster, i, cluster_member, dataset, gt, GD_user, GD_item, GP_user, GP_item, LD_user, LD_item, LP_user, LP_item, opts.topk)
    evaluate_base_emb("GD ", GD_user, GD_item, dataset, idx_cluster, i, cluster_member, gt, opts.topk)
    evaluate_base_emb("GP ", GP_user, GP_item, dataset, idx_cluster, i, cluster_member, gt, opts.topk)
    evaluate_base_emb("LD ", LD_user, LD_item, dataset, idx_cluster, i, cluster_member, gt, opts.topk)
    evaluate_base_emb("LP ", LP_user, LP_item, dataset, idx_cluster, i, cluster_member, gt, opts.topk)