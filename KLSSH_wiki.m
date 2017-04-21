
% Reference:
% Xiangpeng Li
% "Kernel based Latent Semantic Sparse Hashing for Large-scale Retrieval from Heterogeneous Data Sources"
% Neurocomputing
%


clear;
clc;
m = 300;    % number of anchor points
load rand_wiki;
% m=1000;

fid=fopen('klssh_wiki.txt','a+');
trn = 1800; % number of labeled training samples
label_index=randsample(2149,trn)';
sample=randsample(2149,m);
% save('label_and_sample','label_index','sample'); 

if matlabpool('size') <=0 
    matlabpool;
end
run = 1;
map = zeros(run,2);
nbits = [16, 32, 64, 128];
bits = 64;
mu = 0.5;
rho = 0.2;
lambda = 1;

% 按行进行0中心化
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

for count=1:10
    fprintf(fid,'the %d iteration\r\n',count);

for b = 1 : 4
    tic;
    bits = nbits(1, b);
    fprintf('bits = %d, rho = %.4f, lambda = %.4f, mu = %.4f\r\n', bits, rho, lambda, mu);

    for i = 1 : run

    % construct training set

        I_temp = I_tr';
        T_temp = T_tr';
        [row, col]= size(I_temp);
        [rowt, colt] = size(T_temp);

        I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
        T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
        Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
        Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';       
        
        opts = [];
        opts.mu = mu;
        opts.rho = rho;
        opts.lambda = lambda;
        opts.maxOutIter = 4;
        [B,PX,PT,R,A,S,opts]= solveLSSH(I_temp',T_temp',bits,opts);
                
        fprintf('using LSSHcoding method: \n');
        [test_code_Im1,test_code_Te1, train_code1] = LSSHcoding(A, B,PX, PT,R,I_temp',T_temp',Im_te,Te_te,opts, bits);       
        [train_hash1,test_hash_I1,test_hash_T1]=modified_ksh(test_code_Im1,test_code_Te1, train_code1,L_tr,bits,label_index,sample,trn,m);
        
        sim = train_hash1 * test_hash_I1;
        MAP_im = mAP(sim,L_tr,L_te, 0);
        fprintf(fid,'image to text: %.4f\n', MAP_im);

        
        sim = train_hash1 * test_hash_T1;
        MAP_te = mAP(sim,L_tr,L_te, 0);
        fprintf(fid,'text to image: %.4f\r\n', MAP_te);
    end
    toc;

end
end

fclose(fid);
if matlabpool('size') > 0 
    matlabpool close;
end