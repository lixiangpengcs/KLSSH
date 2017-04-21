function [Y,tY_im,tY_te]=midified_ksh(test_code_Im,test_code_Te, train_code,L_tr,bits,label_index,sample,trn,m)


traindata=train_code';
traingnd=L_tr;
testdata_im=test_code_Im';
testdata_te=test_code_Te';

[n,d] = size(traindata);



r = bits;     % number of hash bits

% load label_index_2k; % indexes of labeled samples
% load sample_300;     % indexes of anchors


%% Kernel-Based Supervised Hashing (KSH)
% kernel computing 
tic
anchor = traindata(sample,:);
KTrain = sqdist(traindata',anchor');
sigma = mean(mean(KTrain,2));
KTrain = exp(-KTrain/(2*sigma));
mvec = mean(KTrain);
KTrain = KTrain-repmat(mvec,n,1);

% pairwise label matrix 求两个instance的相似性
trngnd = traingnd(label_index');
temp = repmat(trngnd,1,trn)-repmat(trngnd',trn,1);
S0 = -ones(trn,trn);
tep = find(temp == 0);
S0(tep) = 1;
clear temp;
clear tep;
S = r*S0;

% projection optimization
KK = KTrain(label_index',:);
RM = KK'*KK; 
A1 = zeros(m,r);
flag = zeros(1,r);
for rr = 1:r
    if rr > 1
        S = S-y*y';
    end
    
    LM = KK'*S*KK;
    [U,V] = eig(LM,RM);%求特征值和特征向量，U为特征向量，V为特征值的集合
    eigenvalue = diag(V)';
    [eigenvalue,order] = sort(eigenvalue,'descend');
    A1(:,rr) = U(:,order(1));%最大特征向量
    tep = A1(:,rr)'*RM*A1(:,rr);%l
    A1(:,rr) = sqrt(trn/tep)*A1(:,rr);
    clear U;    
    clear V;
    clear eigenvalue; 
    clear order; 
   % clear tep;  
    
    [get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
    y = double(KK*A1(:,rr)>0);
    ind = find(y <= 0);
    y(ind) = -1;
    clear ind;
    y1 = double(KK*get_vec>0);
    ind = find(y1 <= 0);
    y1(ind) = -1;
    clear ind;
    if y1'*S*y1 > y'*S*y
        flag(rr) = 1;
        A1(:,rr) = get_vec;
        y = y1;
    end
end

% encoding
Y = single(A1'*KTrain' > 0);
tep = find(Y<=0);
Y(tep) = -1;
train_time = toc;
Y=Y';
[train_time]
clear tep; 
clear get_vec;
clear y;
clear y1;
clear S;
clear KK;
clear LM;
clear RM;

%% test
% encoding
tn_im = size(testdata_im,1);
KTest_im = sqdist(testdata_im',anchor');
KTest_im = exp(-KTest_im/(2*sigma));
KTest_im = KTest_im-repmat(mvec,tn_im,1);
tY_im = single(A1'*KTest_im' > 0);
tep_im = find(tY_im<=0);
tY_im(tep_im) = -1;
clear tep;

tn_te = size(testdata_te,1);
KTest_te = sqdist(testdata_te',anchor');
KTest_te = exp(-KTest_te/(2*sigma));
KTest_te = KTest_te-repmat(mvec,tn_te,1);
tY_te = single(A1'*KTest_te' > 0);
tep_te = find(tY_te<=0);
tY_te(tep_te) = -1;
clear tep;
