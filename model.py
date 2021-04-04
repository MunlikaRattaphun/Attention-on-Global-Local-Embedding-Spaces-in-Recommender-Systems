# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:43:36 2020

@author: Kero
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras.backend as K
    
def eu_distance_loss(user_blend, pitem_blend, nitem_blend):

    margin = 0.5
    positive_distance = K.square(user_blend - pitem_blend)
    negative_distance = K.square(user_blend - nitem_blend)
    loss = positive_distance - negative_distance + margin
    loss = K.maximum(0.0, loss)
    loss = K.mean(loss)
    
    return loss

def dot_product_loss(user_blend, pitem_blend, nitem_blend):
    
    loss = 1.0 - K.sigmoid(K.sum(user_blend * pitem_blend) - K.sum(user_blend * nitem_blend))
    loss = K.mean(loss)
    
    return loss

def ATT_AD(GD_user, GP_user, LD_user, LP_user, GD_pitem, GP_pitem, LD_pitem, LP_pitem, GD_nitem, GP_nitem, LD_nitem, LP_nitem, dim):

    #-------------------------------- input layer -------------------------------------
    GD_u = keras.Input(shape=(64,), name = 'user_input1')   
    GD_p = keras.Input(shape=(64,), name = 'pitem_input1')   
    GD_n = keras.Input(shape=(64,), name = 'nitem_input1')   

    GP_u = keras.Input(shape=(64,), name = 'user_input2')   
    GP_p = keras.Input(shape=(64,), name = 'pitem_input2')   
    GP_n = keras.Input(shape=(64,), name = 'nitem_input2')   

    LD_u = keras.Input(shape=(64,), name = 'user_input3')
    LD_p = keras.Input(shape=(64,), name = 'pitem_input3')
    LD_n = keras.Input(shape=(64,), name = 'nitem_input3')

    LP_u = keras.Input(shape=(64,), name = 'user_input4')
    LP_p = keras.Input(shape=(64,), name = 'pitem_input4')
    LP_n = keras.Input(shape=(64,), name = 'nitem_input4')
    
    # ------------------------------ hidden leyer  ------------------------------------
    # layer1   
    h1_f1 = layers.Dense(dim, activation='relu', name = 'h1_f1') 
    GD_q1 = h1_f1(GD_u)
    GD_p1 = h1_f1(GD_p)    

    h1_f2 = layers.Dense(dim, activation='relu', name = 'h1_f2') 
    GP_q2 = h1_f2(GP_u)
    GP_p2 = h1_f2(GP_p)

    h1_f3 = layers.Dense(dim, activation='relu', name = 'h1_f3') 
    LD_q3 = h1_f3(LD_u)
    LD_p3 = h1_f3(LD_p)    
    
    h1_f4 = layers.Dense(dim, activation='relu', name = 'h1_f4') 
    LP_q4 = h1_f4(LP_u)
    LP_p4 = h1_f4(LP_p)
    
    # layer2 
    h2_f1 = layers.Dense(dim, activation='relu', name = 'h2_f1')
    GD_q1 = h2_f1(GD_q1)
    GD_p1 = h2_f1(GD_p1) 
    
    h2_f2 = layers.Dense(dim, activation='relu', name = 'h2_f2')
    GP_q2 = h2_f2(GP_q2)
    GP_p2 = h2_f2(GP_p2)
    
    h2_f3 = layers.Dense(dim, activation='relu', name = 'h2_f3')
    LD_q3 = h2_f3(LD_q3)
    LD_p3 = h2_f3(LD_p3)      
    
    h2_f4 = layers.Dense(dim, activation='relu', name = 'h2_f4')
    LP_q4 = h2_f4(LP_q4)
    LP_p4 = h2_f4(LP_p4)

    # layer3
    h3_f1 = layers.Dense(dim, activation='relu', name = 'h3_f1')
    GD_q1 = h3_f1(GD_q1)
    GD_p1 = h3_f1(GD_p1) 

    h3_f2 = layers.Dense(dim, activation='relu', name = 'h3_f2')
    GP_q2 = h3_f2(GP_q2)
    GP_p2 = h3_f2(GP_p2)

    h3_f3 = layers.Dense(dim, activation='relu', name = 'h3_f3')
    LD_q3 = h3_f3(LD_q3)
    LD_p3 = h3_f3(LD_p3)  
    
    h3_f4 = layers.Dense(dim, activation='relu', name = 'h3_f4')
    LP_q4 = h3_f4(LP_q4)
    LP_p4 = h3_f4(LP_p4)
    
    # softmax layer
    GD_score =  tf.keras.layers.Dot(axes=1)([GD_q1, GD_p1])
    GP_score =  tf.keras.layers.Dot(axes=1)([GP_q2, GP_p2])
    LD_score =  tf.keras.layers.Dot(axes=1)([LD_q3, LD_p3])
    LP_score =  tf.keras.layers.Dot(axes=1)([LP_q4, LP_p4])
    concat = tf.keras.layers.Concatenate(name = 'concat_layer')([GD_score, GP_score, LD_score, LP_score])
    alpha = layers.Dense(4, activation='softmax', name = "alpha")(concat)

    # --------------------------- blending function --------------------------------  
    # users
    GDuser_blend = tf.keras.layers.multiply([alpha[:, 0], GD_u])
    GPuser_blend = tf.keras.layers.multiply([alpha[:, 1], GP_u])
    LDuser_blend = tf.keras.layers.multiply([alpha[:, 2], LD_u])
    LPuser_blend = tf.keras.layers.multiply([alpha[:, 3], LP_u])
    user_blend = tf.keras.layers.Add()([GDuser_blend, GPuser_blend, LDuser_blend, LPuser_blend]) 

    # item positive
    GDpitem_blend = tf.keras.layers.multiply([alpha[:, 0], GD_p])
    GPpitem_blend = tf.keras.layers.multiply([alpha[:, 1], GP_p])
    LDpitem_blend = tf.keras.layers.multiply([alpha[:, 2], LD_p])
    LPpitem_blend = tf.keras.layers.multiply([alpha[:, 3], LP_p])
    pitem_blend = tf.keras.layers.Add()([GDpitem_blend, GPpitem_blend, LDpitem_blend, LPpitem_blend]) 

    # item negative        
    GDnitem_blend = tf.keras.layers.multiply([alpha[:, 0], GD_n])
    GPnitem_blend = tf.keras.layers.multiply([alpha[:, 1], GP_n])
    LDnitem_blend = tf.keras.layers.multiply([alpha[:, 2], LD_n])
    LPnitem_blend = tf.keras.layers.multiply([alpha[:, 3], LP_n])
    nitem_blend = tf.keras.layers.Add()([GDnitem_blend, GPnitem_blend, LDnitem_blend, LPnitem_blend])    
      
    model = tf.keras.models.Model(inputs = [GD_u, GP_u, LD_u, LP_u, GD_p, GP_p, LD_p, LP_p, GD_n, GP_n, LD_n, LP_n], outputs=[user_blend, pitem_blend, nitem_blend])
    model.add_loss(eu_distance_loss(user_blend, pitem_blend, nitem_blend))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00017), loss=None)
    return model

def ATT_AP(GD_user, GP_user, LD_user, LP_user, GD_pitem, GP_pitem, LD_pitem, LP_pitem, GD_nitem, GP_nitem, LD_nitem, LP_nitem, dim):

    #-------------------------------- input layer -------------------------------------
    GD_u = keras.Input(shape=(64,), name = 'user_input1')   
    GD_p = keras.Input(shape=(64,), name = 'pitem_input1')   
    GD_n = keras.Input(shape=(64,), name = 'nitem_input1')   

    GP_u = keras.Input(shape=(64,), name = 'user_input2')   
    GP_p = keras.Input(shape=(64,), name = 'pitem_input2')   
    GP_n = keras.Input(shape=(64,), name = 'nitem_input2')   

    LD_u = keras.Input(shape=(64,), name = 'user_input3')
    LD_p = keras.Input(shape=(64,), name = 'pitem_input3')
    LD_n = keras.Input(shape=(64,), name = 'nitem_input3')

    LP_u = keras.Input(shape=(64,), name = 'user_input4')
    LP_p = keras.Input(shape=(64,), name = 'pitem_input4')
    LP_n = keras.Input(shape=(64,), name = 'nitem_input4')
    
    # ------------------------------ hidden leyer  ------------------------------------
    # layer1   
    h1_f1 = layers.Dense(dim, activation='relu', name = 'h1_f1') 
    GD_q1 = h1_f1(GD_u)
    GD_p1 = h1_f1(GD_p)    

    h1_f2 = layers.Dense(dim, activation='relu', name = 'h1_f2') 
    GP_q2 = h1_f2(GP_u)
    GP_p2 = h1_f2(GP_p)

    h1_f3 = layers.Dense(dim, activation='relu', name = 'h1_f3') 
    LD_q3 = h1_f3(LD_u)
    LD_p3 = h1_f3(LD_p)    
    
    h1_f4 = layers.Dense(dim, activation='relu', name = 'h1_f4') 
    LP_q4 = h1_f4(LP_u)
    LP_p4 = h1_f4(LP_p)
    
    # layer2 
    h2_f1 = layers.Dense(dim, activation='relu', name = 'h2_f1')
    GD_q1 = h2_f1(GD_q1)
    GD_p1 = h2_f1(GD_p1) 
    
    h2_f2 = layers.Dense(dim, activation='relu', name = 'h2_f2')
    GP_q2 = h2_f2(GP_q2)
    GP_p2 = h2_f2(GP_p2)
    
    h2_f3 = layers.Dense(dim, activation='relu', name = 'h2_f3')
    LD_q3 = h2_f3(LD_q3)
    LD_p3 = h2_f3(LD_p3)      
    
    h2_f4 = layers.Dense(dim, activation='relu', name = 'h2_f4')
    LP_q4 = h2_f4(LP_q4)
    LP_p4 = h2_f4(LP_p4)

    # layer3
    h3_f1 = layers.Dense(dim, activation='relu', name = 'h3_f1')
    GD_q1 = h3_f1(GD_q1)
    GD_p1 = h3_f1(GD_p1) 

    h3_f2 = layers.Dense(dim, activation='relu', name = 'h3_f2')
    GP_q2 = h3_f2(GP_q2)
    GP_p2 = h3_f2(GP_p2)

    h3_f3 = layers.Dense(dim, activation='relu', name = 'h3_f3')
    LD_q3 = h3_f3(LD_q3)
    LD_p3 = h3_f3(LD_p3)  
    
    h3_f4 = layers.Dense(dim, activation='relu', name = 'h3_f4')
    LP_q4 = h3_f4(LP_q4)
    LP_p4 = h3_f4(LP_p4)
    
    # softmax layer
    GD_score =  tf.keras.layers.Dot(axes=1)([GD_q1, GD_p1])
    GP_score =  tf.keras.layers.Dot(axes=1)([GP_q2, GP_p2])
    LD_score =  tf.keras.layers.Dot(axes=1)([LD_q3, LD_p3])
    LP_score =  tf.keras.layers.Dot(axes=1)([LP_q4, LP_p4])
    concat = tf.keras.layers.Concatenate(name = 'concat_layer')([GD_score, GP_score, LD_score, LP_score])
    alpha = layers.Dense(4, activation='softmax', name = "alpha")(concat)

    # --------------------------- blending function --------------------------------  
    # users
    GDuser_blend = tf.keras.layers.multiply([alpha[:, 0], GD_u])
    GPuser_blend = tf.keras.layers.multiply([alpha[:, 1], GP_u])
    LDuser_blend = tf.keras.layers.multiply([alpha[:, 2], LD_u])
    LPuser_blend = tf.keras.layers.multiply([alpha[:, 3], LP_u])
    user_blend = tf.keras.layers.Add()([GDuser_blend, GPuser_blend, LDuser_blend, LPuser_blend]) 

    # item positive
    GDpitem_blend = tf.keras.layers.multiply([alpha[:, 0], GD_p])
    GPpitem_blend = tf.keras.layers.multiply([alpha[:, 1], GP_p])
    LDpitem_blend = tf.keras.layers.multiply([alpha[:, 2], LD_p])
    LPpitem_blend = tf.keras.layers.multiply([alpha[:, 3], LP_p])
    pitem_blend = tf.keras.layers.Add()([GDpitem_blend, GPpitem_blend, LDpitem_blend, LPpitem_blend]) 

    # item negative        
    GDnitem_blend = tf.keras.layers.multiply([alpha[:, 0], GD_n])
    GPnitem_blend = tf.keras.layers.multiply([alpha[:, 1], GP_n])
    LDnitem_blend = tf.keras.layers.multiply([alpha[:, 2], LD_n])
    LPnitem_blend = tf.keras.layers.multiply([alpha[:, 3], LP_n])
    nitem_blend = tf.keras.layers.Add()([GDnitem_blend, GPnitem_blend, LDnitem_blend, LPnitem_blend])    
      
    model = tf.keras.models.Model(inputs = [GD_u, GP_u, LD_u, LP_u, GD_p, GP_p, LD_p, LP_p, GD_n, GP_n, LD_n, LP_n], outputs=[user_blend, pitem_blend, nitem_blend])
    model.add_loss(dot_product_loss(user_blend, pitem_blend, nitem_blend))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00017), loss=None)
    return model
