function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision]=mlknn(train_data,train_target,test_data,test_target)
Num=10;
Smooth=1;
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
