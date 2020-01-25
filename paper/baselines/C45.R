library(partykit)
library(RWeka)

root = 'C:/Py-projects/HBL-single/main/sbrl_train/juvenile_new/'
acc = 0
files = 1

for (i in 1:5){

  fold = paste('fold_',as.character(i),'/',sep='')
  
  data_train = read.csv(paste(root,fold,'train_bin.csv',sep=''))
  data_test = read.csv(paste(root,fold,'test_bin.csv',sep=''))
  
  for (name in names(data_train)) {
    data_train[name] = as.factor(data_train[,name])
    data_test[name] = as.factor(data_test[,name])
  }
  
  names(data_train)[length(names(data_train))] = 'group'
  names(data_test)[length(names(data_test))] = 'group'
  
  
  ctree<-J48(group~.,data=data_train,control=Weka_control(C=0.1, M=40, U=FALSE))
  pretree<-predict(ctree,newdata=data_test)
  print(ctree)
  #plot(ctree,type="simple")
  #table(pretree,data_test$group,dnn=c("预测值","真实值"))
  print(sum(pretree==data_test$group)/dim(data_test)[1])
  acc = acc + sum(pretree==data_test$group)/dim(data_test)[1]
}

print(acc/5)


