function [test_code_Im,test_code_Te, train_code] = LSSHcoding(A,B,PX, PT,R,Xtraining, Ttraining, Xtest, Ttest, opts, hash_bits)

%
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com), Yuchen Guo (yuchen.w.guo@gmail.com)
%

train_data_X = preprocessingData(Xtraining',PX,mean( (Xtraining'),2));
train_data_T = preprocessingData(Ttraining',PT,mean( (Ttraining'),2));

test_data_X = preprocessingData(Xtest',PX,mean( (Xtraining'),2));
test_data_T = preprocessingData(Ttest',PT,mean( (Ttraining'),2));

test_code_X = zeros(size(B,2),size(test_data_X,2));
train_code_X = zeros(size(B,2),size(train_data_X,2));


rho = opts.rho;
parfor i = 1 : size(train_data_X,2)
    train_code_X(:,i) =  LeastR(B, train_data_X(:,i), rho, opts);
end
parfor i = 1 : size(test_data_X,2)
    test_code_X(:,i) =  LeastR(B, test_data_X(:,i), rho, opts);
end

test_code_Im=R*test_code_X;
% save('test_code_Im','test_code_Im');

test_code_Te=(A'*A + opts.lambda/opts.mu*eye(size(A,2)))\A'*test_data_T;
% save('test_code_Te','test_code_Te');

train_code=(A'*A + opts.lambda/opts.mu *eye(hash_bits))\(opts.lambda/opts.mu*R*train_code_X + A'*train_data_T);
% save('train_code','train_code');