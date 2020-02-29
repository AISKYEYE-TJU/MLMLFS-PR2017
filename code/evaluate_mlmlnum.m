function  [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision]=evaluate_mlmlnum(classifier,fea_ind,X_train,Train_target,X_test,Test_target)


if length(fea_ind)>100
    num=100;
else
    num=length(fea_ind);
end
    
    
for i=1:num
  switch classifier
    case 'MLKNN'
        [HammingLoss(i),RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)]=mlknn(X_train(:,fea_ind(1:i)),Train_target',X_test(:,fea_ind(1:i)),Test_target');
      case 'RankSVM'
         svm.type='RBF';
         svm.para=2;
         [Weights,Bias,SVs,Weights_sizepre,Bias_sizepre,svm_used,iteration] = RankSVM_train(X_train(:,fea_ind(1:i)),Train_target',svm);
        [HammingLoss(i),RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)]=RankSVM_test(X_test(:,fea_ind(1:i)),Test_target',svm,Weights,Bias,SVs,Weights_sizepre,Bias_sizepre);
  end

end

