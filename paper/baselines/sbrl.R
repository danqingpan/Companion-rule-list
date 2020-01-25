library(sbrl)

root = 'C:/Py-projects/HBL-single/main/sbrl_train/juvenile_new/'
files = 5


for (i in files:files){
  
  fold = paste('fold_',as.character(i),'/',sep='')
  
  #data = read.csv(paste(root,fold,'train_bin.csv',sep=''))
  
  data_train = read.csv(paste(root,fold,'train_bin.csv',sep=''))
  data_test = read.csv(paste(root,fold,'test_bin.csv',sep=''))

  for (name in names(data_train)) {
    data_train[name] = as.factor(data_train[,name])
    data_test[name] = as.factor(data_test[,name])
  }
  
  names(data_train)[length(names(data_train))] = 'label'
  names(data_test)[length(names(data_test))] = 'label'
   
  
  sbrl_model = sbrl(data_train,iters=100000,pos_sign='1',neg_sign='0',rule_minlen = 1,
                    minsupport_pos = 0.3,minsupport_neg = 0.3,rule_maxlen=2,lambda=1,nchain=5,eta = 1.0)
  
  yhat = predict(sbrl_model,data_test)
  write.csv(yhat[1],file=paste(root,fold,'sbrl.csv',sep=''),row.names = FALSE)

}

print(sbrl_model)