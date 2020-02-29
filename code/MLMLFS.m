function [ W, bt, Y_output ] = MLMLFS( X_train,  Y_train, Y_original, L, alpha, lamda, p )
%% 
% The function is to sove the problem of Multi-label Feature Selection with Missing Labels (MLMLFS)
%
% Input:
%   X_train: feature matrix d*n, where d is feature dimension, n is sample number
%   Y_train: label matrix n*c, where n is sample number, c is class number
%   L: Laplacian matrix
%   alpha: parameter
%   lamda: parameter 
%   p: L2p-norm regularization, default value {0.4,0.6,0.8,1.0}

% Output:
%   W: feature selection matrix d*c
%   bt: bias term
%   Y_output: the complete label matrix after label recovery n*c

% Reference: 
% [1] Zhu P, Xu Q, Hu Q, Zhang C. Robust Multi-label Feature Selection with Missing Labels.
%%

[dim, n] = size(X_train);
label_num = size(Y_train, 2);

W = rand(dim, label_num); % Initialize W
bt = rand(label_num, 1);  % Initialize bt

T = X_train' * W + ones(n, 1) * bt' - Y_train;
G0 = diag(0.5./sqrt(sum(T.*T, 2) + eps));
G2 = eye(dim);

mm = sum(sum(G0)); 
nn= sum(L); 
m = sum(sum(G0)) + alpha * sum(sum(L));   
H = eye(n) - (1/m) * ones(n,n) * G0;      
N = eye(n) - (1/m) * ones(n,n) * (G0 + alpha * L);  

%% initialization of Ypred
given_locations = (Y_train ~= 0);
Ypred = rand(n, label_num);   
Y_output = rand(n, label_num);
Ypred(given_locations) = Y_original(given_locations); 
Ypred = sparse(Ypred); 

iter = 1;
obji = 1;

while 1
%% update W
        if dim < n  
            W = pinv(X_train * N' * (G0 + alpha * L) * N * X_train' + lamda * G2) * X_train * N' * (G0 * H + alpha * L *(H - 1)) * Y_train; 
            
            wc = (sum(W .* W, 2) + eps).^(1 - p/2);
            Gw = (2/p) * wc;
            Gw = 1./Gw;
            G2 = spdiags(Gw, 0, dim ,dim);  % G2 update
            
            T = X_train' * W + ones(n, 1) * bt' - Y_train;
            G0 = diag(0.5./sqrt(sum(T.*T, 2) + eps));  % G0 update
            
            m = sum(sum(G0)) + alpha * sum(sum(L));   
            H = eye(n) - (1/m) * ones(n,n) * G0;     
            N = eye(n) - (1/m) * ones(n,n) * (G0 + alpha * L); 
            
        else
            K = X_train * N';
            W = pinv(G2) * K * pinv(lamda * eye(n) + (G0 + alpha * L) * K' * pinv(G2) * K) * (G0 * H + alpha * L * (H -1)) * Y_train;
            
            wc = (sum(W .* W, 2) + eps).^(1 - p/2);
            Gw = (2/p) * wc;
            Gw = 1./Gw;
            G2 = spdiags(Gw, 0, dim, dim);
            
            T = X_train' * W + ones(n, 1) * bt' - Y_train;
            G0 = diag(0.5./sqrt(sum(T.*T, 2) + eps)); 
            
            m = sum(sum(G0)) + alpha * sum(sum(L));   
            H = eye(n) - (1/m) * ones(n,n) * G0;       
            N = eye(n) - (1/m) * ones(n,n) * (G0 + alpha * L); 
        end  
    

    bt = (1/m) * Y_train' * G0 * ones(n, 1) - (1/m) * W' * X_train * (G0 + alpha * L') * ones(n, 1); 
    
%% label prediction
 
    Ypred = X_train' * W + ones(n, 1) * bt';
    
    for i = 1:n
        for j = 1:label_num
           if given_locations(i, j) == 0  
               if Ypred(i, j) <= -1
                   Y_train(i, j) = -1;
               else
                   if Ypred(i, j) >= 1
                       Y_train(i, j) = 1;
                   else
                       Y_train(i, j) = Ypred(i ,j);
                   end
               end
           else
               Y_train(i, j) = Y_original(i, j);
           end
        end
    end
  
    E = X_train'*W + ones(n,1) * bt' - Y_train; 
    objective(iter) = sum(sqrt(sum(E.*E,2)+eps)) + alpha * trace ((X_train'*W + ones(n,1) * bt')' * L * (X_train'*W + ones(n,1) * bt')) + lamda * sum(wc);  

    temp_linear(iter) = sum(sqrt(sum(E.*E,2)+eps));
    temp_dep(iter) = trace ((X_train'*W + ones(n,1) * bt')' * L * (X_train'*W + ones(n,1) * bt'));
    temp_con(iter) =  sum(wc);
    
    cver = abs((objective(iter) - obji)/obji);
    fprintf('(iter_value,cver_value)==========(%d,%d)\n',iter, cver);
    obji = objective(iter);
    iter = iter + 1;
   % if (cver < 10^-5 && iter > 2) 
    if (iter == 21)
        break;
    end        
end
fprintf('\n');

%% label value discretization   
   for i = 1:n
       for j = 1:label_num
           if given_locations(i, j) == 0
               if Y_train(i, j) < 0
                   Y_output(i, j) = -1;
               else
                   if Y_train(i, j) > 0
                       Y_output(i, j) = 1;
                   end
               end
           else
               Y_output(i, j) =  Y_original(i, j); 
           end
       end
   end
   
end