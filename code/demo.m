   
    datalist{1}='art';
    datalist{2}='birds';
    datalist{3}='ref';
    datalist{4}='social';
    datalist{5}='yeast';

currentFolder = pwd;
addpath(genpath(currentFolder));
    
method='MLMLFS'
classifier='MLKNN'

lam_ind= -6:2:6;
lam_pool=10.^lam_ind;  

alpha_ind =-6:2:-2; 
alpha_pool = 10.^alpha_ind;

data_select=2;%1:6;  
 
for pi=1:length(lam_pool)
    for k=1:length(alpha_pool)
        alpha = alpha_pool(k); 
        for i=1:length(data_select)
                    kk=data_select(i);
                    eval(['load ' [datalist{kk} '_train']])
                    eval(['load ' [datalist{kk} '_test']])
                    fprintf(datalist{kk});
                    fprintf('(lam_value,alpha_value,data)==========(%d,%d,%d)\n',pi,k,i);
                    
                    lamda=lam_pool(pi);
                    p=1; % p=0.4,0.6,0.8,1  

					L_x = Laplacian(train_data');        
                    [ W, bt, Y_output ] = MLMLFS( train_data', Y_missing', train_target', L_x, alpha, lamda, p); 
                    score= sum(W.*W,2);
                    [~,fea_ind] = sort(score,'descend');  

                    [HammingLoss{i},RankingLoss{i},OneError{i},Coverage{i},Average_Precision{i}]=evaluate_mlmlnum(classifier,fea_ind,train_data,Y_output,test_data,test_target');
        end
      
        lamAlphaDataHam{pi,k}=HammingLoss;
        lamAlphaDataRank{pi,k}=RankingLoss;
        lamAlphaDataOneErr{pi,k}=OneError;
        lamAlphaDataCov{pi,k}=Coverage;
        lamAlphaDataAP{pi,k}=Average_Precision;
    end 
end


for k=1:length(data_select)
    s=0;
    result1=[];
    result2=[];
    result3=[];
    result4=[];
    result5=[];
    
    for m=1:length(lam_pool)
       for n=1:length(alpha_pool)
           s=s+1;
           result1(s)=min(lamAlphaDataHam{m,n}{k});
           result2(s)=min(lamAlphaDataRank{m,n}{k});
           result3(s)=min(lamAlphaDataOneErr{m,n}{k});
           result4(s)=min(lamAlphaDataCov{m,n}{k});
           result5(s)=max(lamAlphaDataAP{m,n}{k});          
       end
    end
    
    value1(k)=min(result1); 
    value2(k)=min(result2);
    value3(k)=min(result3);
    value4(k)=min(result4);
    value5(k)=max(result5);   

end

 result=[value1',value2',value3',value4',value5']

 save result result

