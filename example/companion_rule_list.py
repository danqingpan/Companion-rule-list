# -*- coding: utf-8 -*-
"""
This is the code for paper Companion rule list

@author: DQ
"""


import numpy as np
import pandas as pd 
from fim import fpgrowth 
import itertools
import random
import operator
from scipy.sparse import csc_matrix
import math
from bitarray import bitarray
import argparse


def screen_rules(rules,df,y,N,supp):
    
    # This function screen rules by supporting rate
    
    itemInd = {}
    for i,name in enumerate(df.columns):
        itemInd[name] = int(i)
    
    len_rules = [len(rule) for rule in rules]
    indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
    indptr = list(accumulate(len_rules))
    indptr.insert(0,0) # insert 0 at 0 position/necessary for building csc-matrix
    indptr = np.array(indptr)
    data = np.ones(len(indices))
    ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
    
    mat = np.matrix(df)*ruleMatrix # a matrix of data sum wrt rules
    
    lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
    Z = (mat == lenMatrix).astype(int) # matrix with 0 and 1/'match' rule when 1
    
    Zpos = [Z[i] for i in np.where(np.array(y)>0)][0]
    TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
    
    supp_select = np.where(TP>=supp*sum(y)/100)[0]
    FP = np.array(np.sum(Z,axis = 0))[0] - TP
    p1 = TP.astype(float)/(TP+FP)
    
    supp_select = np.array([i for i in supp_select if p1[i]>np.mean(y)],dtype=np.int32)
    select = np.argsort(p1[supp_select])[::-1][:N].tolist()
    ind = list(supp_select[select])
    rules = [rules[i] for i in ind]
    
    RMatrix = np.array(Z[:,ind]) 
    supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind] # support/number of data covered
    
    return rules, RMatrix, supp, p1[ind], FP[ind]


def accumulate(iterable, func = operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> [1 3 6 10 15]
    # accumulate([1,2,3,4,5], operator.mul) --> [1 2 6 24 120]
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def propose_rule(rule_sequence,premined):
    
    # This function propose a new rule based on the previous rule
    
    premined_rules = premined.copy()
    rule_seq = rule_sequence.copy()
    
    # No rule reuse
    for i in range(len(rule_seq)):
        premined_rules.remove(rule_seq[i])
    
    rand = random.random()
    
    # We use 4 operations to generate a new rule
    
    if (rand < 0.25):
        #print('add')
        if len(premined_rules)>1:
            # randomly choose a rule in premined rules
            rule_to_add = random.sample(premined_rules,1)[0]
            # insert to a random position in the list
            rule_seq.insert(random.randint(0,len(rule_seq)),rule_to_add)
        
    elif (rand < 0.5):
        #print('remove')
        if len(rule_seq)>1: # at least have 2 rules in the list
            # randomly choose a rule from the list
            rule_to_remove = random.sample(rule_seq,1)[0]
            # remove it
            rule_seq.remove(rule_to_remove)

    elif (rand < 0.75):
        #print('swap')
        if len(rule_seq)>1: # at least have 2 rules in the list
            # randomly choose 2 rules in the list
            swap_num = random.sample(list(range(len(rule_seq))),2)
            # swap them
            rule_seq[swap_num[0]],rule_seq[swap_num[1]] = rule_seq[swap_num[1]],rule_seq[swap_num[0]]

    else:
        #print('replace')
        if (len(rule_seq)>0 and len(premined_rules)>1):
            # randomly choose a rule in the list
            replace_num = random.sample(list(range(len(rule_seq))),1)[0]
            # randomly choose a premined rule
            rule_selected = random.sample(premined_rules,1)[0]
            # replace the rule in the list
            rule_seq[replace_num] = rule_selected
            
    return rule_seq


def obj_func(cov,Acc,Alpha,acc_blx,c_blx):
    
    k = len(cov)
    
    # The objective function
    # cov: the cover rate of each rule
    # Acc: the accuracy of each rule
    # acc_blx: black box accuracy
    # c_blx: the cover rate of black box
    
    acc = [sum([Acc[i]*cov[i] for i in range(j+1)])+acc_blx[j]*c_blx[j] for j in range(k)]
    obj_list = [0.5*(acc[i]+acc[i-1])*cov[i] for i in range(1,k)]
    
    obj = sum(obj_list)+0.5*(acc[0]+BLX_ACC)*cov[0]#
    obj = obj - Alpha*k

    return obj


def compute_support(rule,support_map,cover_map,cov_blx,start):
    
    # this function generates support map and cover map by less computation
    # only copy before start
    
    # new rule, previous support, previous cover and a start point
    # start should not be larger than len(rule_list)-1 or len(support_map)
    
    new_support_map = {} # accumulate set
    new_cover_map = {} # non accumulate set
    new_cov_blx = []
    
    # copy before start
    for i in range(start): 
        
        new_support_map[i] = support_map[i]
        new_cover_map[i] = cover_map[i]
        new_cov_blx.append(cov_blx[i])
    
    # compute after start
    for i in range(start,len(rule)): # only do set computation from start

        if i  == 0:
            new_support_map[i] = cover_sets[rule[i]]
            new_cover_map[i] = cover_sets[rule[i]]
        
        else:
            new_support_map[i] = (cover_sets[rule[i]])|(new_support_map[i-1]) # time cost 3
            new_cover_map[i] = (cover_sets[rule[i]])&~(new_support_map[i-1]) # time cost 4
        
        new_cov_blx.append(~new_support_map[i])
        
    return new_support_map, new_cover_map, new_cov_blx


def compute_obj(new_rule_n,prev_rule_n,support_map,cover_map,cov_blx,Y,Alpha,c,A,choose):
    
    # This function compute objective function
    k = len(new_rule_n)
    
    # prev_rule and support_map match
    c_new = [] # cover rate of each rule
    A_new = [] # accuracy of each rule
    choose_new = []

    # find start position by comparing new rule and previous rule

    start_position = compute_start(new_rule_n,prev_rule_n)
    
    new_support_map,new_cover_map,new_cov_blx = compute_support(new_rule_n,support_map,cover_map,cov_blx,start_position)

    # copy before start position
    for i in range(0,start_position):
        c_new.append(c[i])
        A_new.append(A[i])
        choose_new.append(choose[i])
        
    # compute after start position
    for i in range(start_position,k):
        
        pos_label_num = sum(Y&(new_cover_map[i])) # time cost 1
        covered_num = sum(new_cover_map[i])
                
        pos_acc = pos_label_num/(covered_num+0.0001)
        neg_acc = (covered_num-pos_label_num)/(covered_num+0.0001)
        
        c_new.append(covered_num/df.shape[0])
        
        #A_new.append(max(pos_acc,neg_acc))
        if pos_acc > neg_acc:
            A_new.append(pos_acc)
            choose_new.append(1)
        
        else:
            A_new.append(neg_acc)
            choose_new.append(0)
    
    #new_cov_blx = [~new_support_map[i] for i in range(len(new_support_map))]
    blx_cov_num = [sum(new_cov_blx[i]) for i in range(len(new_support_map))]
    
    acc_blx = [sum(new_cov_blx[i]&Y_c)/(blx_cov_num[i]+0.0001) for i in range(len(new_support_map))]
    c_blx = [blx_cov_num[i]/df.shape[0] for i in range(len(new_support_map))]

    # compute object
    obj = obj_func(c_new,A_new,Alpha,acc_blx,c_blx)
    #print(A_blx_new)
    return obj, c_new, A_new, choose_new, new_support_map, new_cover_map, new_cov_blx


def simulated_annealing(init_rule,init_T,Alpha,iteration):
    
    # The main loop of simulated annealing
    
    # temperature
    temperature = init_T
    
    obj_p,chosen,A = 0.1,[],[]

    obj_best,cover_best,c_best,A_best = 0,0,[],[]
    rule_best,chosen_best = [],[]
    
    # init support map and cover map
    support_map,cover_map,cov_blx = compute_support(init_rule,{},{},[],start=0)

    # init Accuracy, cover and chosen
    c = [sum(cover_map[i])/df.shape[0] for i in range(len(init_rule))]
    
    for i in range(len(init_rule)):
        init_pos_acc = sum(Y&(cover_map[i]))/(sum(cover_map[i])+0.001)
        init_neg_acc = (sum(cover_map[i])-sum(Y&(cover_map[i])))/(sum(cover_map[i])+0.001)
        
        if init_pos_acc>init_neg_acc:
            A.append(init_pos_acc)
            chosen.append(1)
        else:
            A.append(init_neg_acc)
            chosen.append(0)
    
    prev_rule = init_rule
    
    # main iteration
    for t in range(iteration):
        
        rule_proposed = propose_rule(prev_rule,list(range(len(premined_rules))))
        obj_c, c_new, A_new, chosen_new,new_support_map, new_cover_map,new_cov_blx  = compute_obj(rule_proposed,prev_rule,support_map,cover_map,cov_blx,Y,Alpha,c,A,chosen)
        
        if obj_c > obj_p:
            #print('accept directly')
            obj_p,prev_rule,c,A,support_map,cover_map,cov_blx,chosen = obj_c,rule_proposed,c_new,A_new,new_support_map,new_cover_map,new_cov_blx,chosen_new
            
            # must check again here
            if obj_c > obj_best:
                
                obj_best, c_best,A_best,rule_best,chosen_best= obj_c, c_new,A_new,rule_proposed,chosen_new
                cover_best = new_support_map[len(new_support_map)-1]

        else:
            rand = random.random()
            accept_rate = math.exp((obj_c-obj_p)/temperature)
            
            if accept_rate > rand:
                
                obj_p,prev_rule,c,A,support_map,cover_map,cov_blx,chosen = obj_c,rule_proposed,c_new,A_new,new_support_map,new_cover_map,new_cov_blx,chosen_new

        temperature = init_T/math.log(2+t)

        #other options for updating temperature
        #temperature = temperature*0.999
        #temperature = init_T**(t/iteration)
        
        if  t%1000==0:
            print('step :', t)
            print('cover rate: ',sum(new_support_map[len(new_support_map)-1])/df.shape[0])
            print('rule number: ',len(rule_best))
            print('best obj :', obj_best)
    
    return obj_best, c_best, A_best, cover_best, chosen_best, rule_best


def compute_start(new_rule,prev_rule):
    
    # This function is to find where to start the computation
    # We use this mechanism to speed up the algorithm
    # before start, it is copy
    
    start = 0
    match = False # to check if two rule list have different rule
    len_min = min(len(new_rule),len(prev_rule))
    
    for i in range(len_min):
        if prev_rule[i] != new_rule[i]:
            start = i
            match = True
            break    
    
    if match == False:
        # if no different rule, then the last rule must be removed/added
        start = max(len(new_rule),len(prev_rule))-1
    
    return start


def gen_rule(df_combine,Y,Supp,Maxlen,N):
    
    # generate rules using FP-growth algorithm
    
    df_combine = 1 - df_combine
    
    itemMatrix = [[item for item in df_combine.columns if row[item] == 1] for i,row in df_combine.iterrows()]
    
    pindex = np.where(Y==1)[0] 
    nindex = np.where(Y!=1)[0]
    
    prules = fpgrowth([itemMatrix[i] for i in pindex],supp=Supp,zmin=1,zmax=Maxlen)
    prules = [np.sort(x[0]).tolist() for x in prules]
    
    nrules = fpgrowth([itemMatrix[i] for i in nindex],supp=Supp,zmin=1,zmax=Maxlen)
    nrules = [np.sort(x[0]).tolist() for x in nrules]   

    prules, pRMatrix, psupp, pprecision, perror = screen_rules(prules,df_combine,Y,N,Supp)
    nrules, nRMatrix, nsupp, nprecision, nerror = screen_rules(nrules,df_combine,1-np.array(Y),N,Supp)
    
    premined_rules = prules
    premined_rules.extend(nrules)
    
    return premined_rules
    

def gen_blx_pred(df_part_bin,train_num,blx_train_num,blx_model):
    
    # This function is used to train blx models
    # not used
    
    df_part_bin = np.array(df_part_bin)
    blx_train_data = df_part_bin[train_num:train_num+blx_train_num,:-1]
    blx_train_label = df_part_bin[train_num:train_num+blx_train_num,-1]
    blx_test_data = df_part_bin[:train_num,:-1]
    
    blx_model.fit(blx_train_data,blx_train_label)
    Yb = blx_model.predict(blx_test_data)
    
    return Yb


def mine_rule(maxlen,supp,n,num_data_mine,flag):
    
    # mine rules
    # use flag to choose between mining or using previouly mined rules
    
    # use max num_data_mine observations to mine rules
    rule_mining_data = df.shape[0] if df.shape[0]<num_data_mine else num_data_mine
    
    if flag:
        premined_rules = gen_rule(df.iloc[:rule_mining_data,:],Y_train[:rule_mining_data],Supp=supp,Maxlen=maxlen,N=n)
        np.save('premined_rules.npy',np.array(premined_rules))

    else:
        premined_rules = np.load('premined_rules.npy')
        premined_rules = premined_rules.tolist()
    
    return premined_rules


def test(test_data,test_label,test_Yb,chosen,rule):

    # use test data to test the rule list
    output_rules = [premined_rules[rule[i]] for i in range(len(rule))]
    catch_list = np.array([-1]*test_data.shape[0])
    
    # to show observation is caught by which rule
    for i in range(test_data.shape[0]):
        for j in range(len(output_rules)):
            match = True
            for condition in output_rules[j]:
                if test_data.iloc[i][condition] == 0:
                    match = False
                    break
            if match == True:
                catch_list[i] = j
                break
                
    test_cover_rate = [0]*len(output_rules)
    blx_cover_rate = [0]*len(output_rules)
    test_acc = []
    blx_acc = []
    
    rule_coverd_set = set()
    blx_cover_set = set(range(test_data.shape[0]))
    
    for i in range(len(output_rules)):
        # observation num caught by rule i
        rule_catch = np.where(catch_list==i)[0]
        # the accumulated rules catch by rule list
        rule_coverd_set = rule_coverd_set.union(set(rule_catch))
        # the left part is then caught by blx model
        blx_cover = blx_cover_set.difference(rule_coverd_set)
        # blx cover rate
        blx_cover_rate[i] = len(blx_cover)/(test_data.shape[0]+0.0001)
        # blx accuracy
        blx_acc.append(sum(test_Yb[list(blx_cover)]==test_label[list(blx_cover)])/(len(blx_cover)+0.0001))
        # cover rate and accuracy of rules
        test_cover_rate[i] = len(rule_catch)/(test_data.shape[0]+0.0001)
        test_acc.append(sum(test_label[rule_catch] == chosen[i])/len(rule_catch))
    
    # the overall accuracy of hybrid models
    test_overall_acc = [sum([test_cover_rate[i]*test_acc[i] for i in range(j+1)])+blx_cover_rate[j]*blx_acc[j] for j in range(len(output_rules))]
    
    return test_overall_acc,output_rules,test_cover_rate,test_acc


if __name__ == '__main__':
    #-------------------------------setting----------------------------------------
    # data for train
    
    parser = argparse.ArgumentParser()
    # train data, last column is label
    parser.add_argument("--file",help = 'binarized file', default = 'adult.csv')
    # blx prediction of the same observations
    parser.add_argument("--blx_file", help = 'blackbox labels',default = 'rf_adult.csv')
    # alpha to control the rule list length
    parser.add_argument("--alpha",help = 'regularizaition parameter', default = 0.001)
    # number of iterations
    parser.add_argument("--step", help = 'train step', default = 20000)
    # max number of rules
    parser.add_argument("--card", help = 'cardinality',default = 2)
    # minimal support when choosing rules
    parser.add_argument("--supp", help = 'rule mining support', default = 0.05)
    # number of rules used for both position and negative side
    parser.add_argument("--n", help = 'number of rules', default = 200)
    
    args = parser.parse_args()
    
    data_file = args.file
    blx_file = args.blx_file
    
    alpha = args.alpha
    T = args.step
    maxlen = args.card
    supp = args.supp
    n = args.n
    
    # mine rules from num_data_mine observations
    num_data_mine = 10000
    # the ratio of training data, 0.8 represents 80% as training data 20% as test data
    ratio = 0.5
    # mine rules or use saved rules
    flag = 1
    init_temperature = 0.001  # penalty, smaller -> more rles    


    #-------------------------------data preparation-------------------------------
    #root_file =root+'fold_'+str(fold)
    
    df_with_label = pd.read_csv(data_file)
    label = df_with_label.iloc[:,-1]
    
    Yb_non_bit = np.squeeze(np.array(pd.read_csv(blx_file,header=0)))
    
    columns = list(df_with_label.columns)
    
    df_with_label = df_with_label.drop(columns = columns[-1])
    
    columns_to_add = [(columns[i]+'_rev') for i in range(len(columns)-1)]
    
    for i in range(len(columns_to_add)):
        df_with_label[columns_to_add[i]] = 1 - df_with_label[columns[i]]
    
    df_with_label['y'] = label
    
    # set data(ratio) to train
    data_num = df_with_label.shape[0]
    train_num = int(data_num*ratio)
    
    # train data, train labels, train black box prediction
    Y_train = np.squeeze(np.array(df_with_label.iloc[:,-1]))[:train_num]
    df = df_with_label.drop(columns=df_with_label.columns[-1]).iloc[:train_num,:]
    Yb_train = Yb_non_bit[:train_num]
    
    # mining rules
    premined_rules = mine_rule(maxlen,supp,n,num_data_mine,flag)
    
    #-------------------------------bitarray---------------------------------------
    # must have cover_sets,Y,Yb,Y_c
    # bitarray rule cover
    cover_sets = [bitarray(np.sum(df[premined_rules[i]],axis=1)==len(premined_rules[i])) for i in range(len(premined_rules))]
    
    # bitarray label, bitarray black box prediciton
    Y = bitarray(list(Y_train))
    Yb = bitarray(list(Yb_train))
    Y_c = ~(Y^Yb) # to speed up
    
    # overall black box accuracy
    BLX_ACC = sum(Y_c)/len(Y)
    
    #---------------------------------train----------------------------------------
    # randomly choose 3 rules to start
    
    init_rule = random.sample(list(range(len(premined_rules))),3)
    
    # c: cover for each rule, A:accuracy for each rule, chosen: label chosen, a.k.a prediction
    obj, c, A, cover, chosen, rule = simulated_annealing(init_rule,init_temperature,alpha,T)
    
    #---------------------------------test-----------------------------------------
    
    # test data, test label , test black box prediction
    test_data = df_with_label.iloc[train_num:,:-1]
    test_label = np.array(df_with_label.iloc[train_num:,-1])
    test_Yb = np.array(Yb_non_bit)[train_num:]
    
    test_overall_accuracy, output_rules, test_cover_rate,test_acc = test(test_data,test_label,test_Yb,chosen, rule)
    
    cover_rate = list(accumulate(test_cover_rate))
    
    for i in range(len(output_rules)):
        
        if i == 0:
            rule_text = 'if '
        else:
            rule_text = 'else if '
        
        for j in range(len(output_rules[i])):
            
            if output_rules[i][j][-4:] == '_rev':
                rule_text += output_rules[i][j][:-4] + ' = 0 '
            else:
                rule_text += output_rules[i][j] + ' = 1 '
            
            if j != len(output_rules[i])-1:
                rule_text += ' and  '
        
        rule_text = rule_text + ', then y = ' + str(chosen[i])
        rule_text = rule_text + ', transparency = ' + str(cover_rate[i])
        rule_text = rule_text + ', accuracy = ' + str(test_overall_accuracy[i])
        
        print(rule_text)
    
    
        #print('')




