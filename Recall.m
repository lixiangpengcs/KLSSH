function recall=Recall(sim_x,L_tr,L_te,mark)

trn=size(sim_x,2);
Rc = zeros(trn,1);
R = 100;%top K
recallA=zeros(trn,1);
for i=1:trn
   Px=zeros(R,1);
   deltax = zeros(R,1);
   label = L_te(i);
   t=size(L_tr,1);
   
   [~,inxx] = sort(sim_x(:,i),'descend');
   Lx = length([L_tr(inxx(1:R)) == label]);
   for r = 1 : R
       Lrx = length([L_tr(inxx(1:r)) == label]);
       if label == L_tr(inxx(r))
           deltax(r) = 1;
       end
       Px(r) = Lrx/r;
   end

   APx(i) = sum(Px.*deltax);
   

   num=0;
   for m=1:t  
       if L_tr(m)==label
           num=num+1;
       end
   end
   recallA(i)=(sum(Px.*deltax))/num;
end
recall=mean(recallA);