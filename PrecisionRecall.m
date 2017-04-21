load hashcode_labelme_32_Te2Im;
mark=0;


for R=50:50:1000
tn = size(sim,2);
APx = zeros(tn,1);
Re = zeros(tn,1);
recallA=zeros(tn,1);
t=size(L_tr,1);

%L_tr = [L_tr;L_tr];
for i = 1 : tn
    Px = zeros(R,1);
    deltax = zeros(R,1);
    label = L_te(i);
    if mark == 0
        [~,inxx] = sort(sim(:,i),'descend');
    elseif mark == 1
        [~,inxx] = sort(sim(:,i));
    end
    Lx = length([L_tr(inxx(1:R)) == label]);
    for r = 1 : R
        Lrx = length([L_tr(inxx(1:r)) == label]);
        if label == L_tr(inxx(r))
            deltax(r) = 1;
        end
        Px(r) = Lrx/r;
    end
    if Lx ~=0
        APx(i) = sum(Px.*deltax)/Lx;
        Re(i) = sum(Px.*deltax);
    end
    num=0;
   for m=1:t  
       if L_tr(m)==label
           num=num+1;
       end
   end
   recallA(i)=(sum(Px.*deltax))/num;
end
map(R/50) = mean(APx);
recall(R/50)=mean(recallA);

end
map(21)=0.14;
recall(21)=1;
plot(recall,map);