function [coding] = unsuper_FeaCoding(X)
% x: sample*fea
% coding: fea*1

% pca
%[coeff,score,latent] = pca(X');
%coding=coeff(:,1);
%coding=latent/sum(latent);


% mean
a=(mean(X'))';
coding=a./norm(a);

%coding=a;


end

