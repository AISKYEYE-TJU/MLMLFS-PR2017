% function  [fea_ind,Train_target]=evaluate_mlfs(X_train,Train_target, Y_original, X_test,Test_target,method,lambda,tau,p)
function  [fea_ind,Output_target]=evaluate_mlfs(X_train,Train_target, Y_original, X_test,Test_target,method,alpha,lambda,p)
switch method
    case 'mogfs'
      [W1, Output_target,~,~ ] = MOGFS( X_train,Train_target',Y_original,lambda,5 ); % 5为迭代次数
       W=W1(2:end,:);
       bt=W1(1,:);
      score= sum(W.*W,2);
      [~,fea_ind] = sort(score,'descend');
    case 'CSFS'
      [ W, bt, Output_target ] = CSFS( X_train', Train_target,Y_original, lambda);   
      score= sum(W.*W,2);
      [~,fea_ind] = sort(score,'descend');  
      % Train_target;
    case 'MLMLFS'
      [ W, bt, Output_target ] = MLMLFS( X_train', Train_target, Y_original, lambda, p); 
      score= sum(W.*W,2);
      [~,fea_ind] = sort(score,'descend');  
    case 'MLMLFS_dep'
     %% compute L
      L_x = Laplacian(X_train');        
      [ W, bt, Output_target ] = MLMLFS_dependency( X_train', Train_target, Y_original, L_x, alpha, lambda, p); 
      score= sum(W.*W,2);
      [~,fea_ind] = sort(score,'descend');  
    case 'pmu'   
      numB=2;
      numK=size(X_train,2);
      [train target]=trans(X_train,Train_target,numB);
      [fea_ind,value] = pmu(train, target', numK);
      Output_target = Train_target;
    case 'mlnb'    %%%
      [fea_ind]=MLNB(X_train,Train_target');
      Output_target = Train_target;
    case 'mddm_proj'
       X=normal_11(X_train);
       Train_target=trans_10(Train_target);
       feature=size(X_train,2);
       L=kernel_linear(Train_target'); 
       [tmp_lambda,fea_ind,lambda] = mddm_linear(X, L, 'proj', 0.5, feature);
       Output_target = Train_target;
    case 'mddm_spc'
       X=normal_11(X_train);
       Train_target=trans_10(Train_target);
       feature=size(X_train,2);
       L=kernel_linear(Train_target'); 
       [tmp_lambda,fea_ind,lambda] = mddm_linear(X, L, 'spc', 0.5, feature);
       Output_target = Train_target;
    case 'mdmr'
        numB=2;
        numK=size(X_train,2);
        [train target]=trans_11(X_train,Train_target',numB);
        [fea_ind] = MDMR(train,target,numK );
        Output_target = Train_target;
	case 'sfus'
	   paraset=[0.001,0.01,0.1,1,10,100,1000];
       para.rd=0;
       X=X_train';
       Y=Train_target;
	   para.alpha=1;
	   for i=7:length(paraset)
	      para.beta=paraset(i);
		  W = subspace_J21(X,Y,para);
		  score = sum(W.*W,2);
		  [~,fea_indtmp] = sort(score,'descend');
		  fea_vector{i} = fea_indtmp;
		  if length(fea_indtmp)>100
		    [HammingLoss(i),RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)]=mlknn(X_train(:,fea_indtmp(1:100)),Train_target',X_test(:,fea_indtmp(1:100)),Test_target');
		  else
			[HammingLoss(i),RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)]=mlknn(X_train(:,fea_indtmp),Train_target',X_test(:,fea_indtmp),Test_target');
          end
       end 
	   
	   [ans,maxind]=max(Average_Precision);
	   fea_ind = fea_vector{maxind};
       Output_target = Train_target;
	   
end

