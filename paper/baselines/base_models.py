# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 00:34:44 2019

@author: DQ
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#import graphviz 
#import pydotplus
#from sklearn.externals.six import StringIO


save = 1

dataset = 'messidor'
model = 'xg'


def refresh_blx_prediction(dataset,save,model):
    
    #best_predictions = []
    #best_acc = 0
    #best_clf = 0
    
    file = 4
    folds_num = 1
    for i in range(1,2):
        
        if model == 'dc':
            clf = DecisionTreeClassifier(max_depth=10,max_leaf_nodes = 10 ); blx = 'dc'
        
        
        elif model == 'rf':
            clf = RandomForestClassifier(n_estimators=400, max_depth=15,random_state=0,min_samples_split = 12,n_jobs = -1);blx = 'rf'
        elif model == 'ada':
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=2),n_estimators=200,learning_rate = 1,random_state = 0); blx = 'ada'
        elif model == 'xg':
            clf = XGBClassifier(n_estimatores=20,learning_rate=0.08,max_depth=3, min_child_weight=2,nthread=-1); blx = 'xg'
        
        #nodes = 0
        #acc = 0
        #predictions = []
        
        for fold in range(file,file+folds_num):
            
            if dataset == 'juvenile_new':
                train_file = dataset+'/fold_'+str(fold)+'/train_bin.csv'
            else:
                train_file = dataset+'/fold_'+str(fold)+'/train_digit.csv'
            #test_file = 'sbrl_train/'+dataset+'/fold_'+str(fold)+'/test_bin.csv'                
            
            df_train_data = pd.read_csv(train_file) 
            df_train_1 = df_train_data.iloc[:int(0.5*df_train_data.shape[0]),:]
            df_train_2 = df_train_data.iloc[int(0.5*df_train_data.shape[0]):,:]


            #df_test = pd.read_csv(test_file)
            #test_label = df_test.iloc[:,-1]
            #black_box_label = pd.read_csv(dataset+'/fold_'+str(fold)+'/'+blx+'_test.csv')
            
            # train on later half to predict on previous half
            clf.fit(df_train_2.iloc[:,:-1],df_train_2.iloc[:,-1])
            pred_previous_half = clf.predict(df_train_1.iloc[:,:-1])
            
            # train on previous half to predict on later half
            clf.fit(df_train_1.iloc[:,:-1],df_train_1.iloc[:,-1])
            pred_latter_half = clf.predict(df_train_2.iloc[:,:-1])
            
            
            pred = np.concatenate((pred_previous_half,pred_latter_half), axis=0)
            
            print(sum(pred == np.array(df_train_data.iloc[:,-1]))/df_train_data.shape[0])

            if save == 1 and blx is not 'dc': 

                blx_root = dataset+'/fold_'+str(fold)+'/'+blx+'_train.csv'
                df_save = pd.DataFrame(pred)
                
                df_save.columns = ['y']
                #print(df_save)
                df_save.to_csv(blx_root,index=0)
                    
                    #best_df = pd.DataFrame(best_predictions[0])
                    #best_df.columns = ['y']
                    #best_df.to_csv(blx_root,index=0)

            '''
            print('pre_',(sum(test_label==pred)/len(test_label)))
            acc = acc + (sum(test_label==pred)/len(test_label))
            
            if model == 'dc':
               nodes = nodes + clf.tree_.node_count
               print(clf.tree_.node_count)
               #dot_data = StringIO()
               #tree.export_graphviz(clf,out_file = dot_data,
               #      filled=True,rounded=True,feature_names=feature_name,
               #      class_names=target_name,
               #      special_characters=True)
               #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
               #graph.write_pdf("graph.pdf")
               
        
        
        print(str(i)+': ',acc/5)
        
        #if model == 'dc':
            #print('nodes: ', nodes)
        
        if acc > best_acc:
            best_acc = acc
            #best_clf = clf
            best_predictions = predictions
            
    
    print(best_acc)
              
    if save == 1 and blx is not 'dc': 
        for i in range(file,file+1):
            blx_root = dataset+'/fold_'+str(i)+'/'+blx+'_train.csv'
            best_df = pd.DataFrame(best_predictions[0])
            best_df.columns = ['y']
            best_df.to_csv(blx_root,index=0)
            '''


refresh_blx_prediction(dataset,save,model)




