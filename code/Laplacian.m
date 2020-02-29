function L = Laplacian( X )
%LAPLACIAN_�˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


p=size(X,1);
n=size(X,2);

% This part of code constructs the Laplacian Graph
options = [];
%options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5;
%options.WeightMode = 'Binary';
options.WeightMode = 'HeatKernel';
%options.WeightMode = 'FMELaplacian';
W = constructW(X',options);
D = diag(sum(W,1));
L=D-W;

end

