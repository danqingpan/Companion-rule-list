# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:26:54 2019

@author: 81701
"""


from corels import CorelsClassifier
import pandas as pd
import numpy as np


dataset = 'german'
#fold = 3
#curious,lower_bound,dfs,bfs,objective

acc = 0
for fold in range(2,3):
    train_file = 'sbrl_train/'+dataset+'/fold_'+str(fold)+'/train_bin.csv'
    
    train_data = pd.read_csv(train_file)
    X = np.array(train_data.iloc[:,:-1])
    
    features = list(train_data.columns[:-1])
    y = np.array(train_data.iloc[:,-1])
    
    
    C = CorelsClassifier(max_card=2, c=0.0, n_iter=200000,policy='lower_bound',min_support=0.05)
    
    # Fit the model
    C.fit(X, y, features=features)
    
    # Print the resulting rulelist
    #print(C.rl())
    
    test_file = 'sbrl_train/'+dataset+'/fold_'+str(fold)+'/test_bin.csv'
    test_data = pd.read_csv(test_file)
    X_test = np.array(test_data.iloc[:,:-1])
    y_test = np.array(test_data.iloc[:,-1])
    
    # Predict on the training set
    pred = (C.predict(X_test))
    acc_single = (sum(y_test == pred)/y_test.shape[0])
    print(acc_single)
    acc = acc + acc_single

print('accuracy: ',acc/1)
