function [ m,C,pi,pk_ave ] = EM( m,N,K,X,Z )
max_iter = 20;
C = zeros(2,2,K);
for i = 1:K
    C(:,:,i) = eye(2);
end
pi = 1/K*ones(K,1);

for it = 1:max_iter
    %% E-step
    pk = zeros(N,K);
    for i = 1:N
        x = X(i,:);
        piNormDist = zeros(K,1);
        for k = 1:K
            piNormDist(k) = pi(k)*mvnpdf(x,m(k,:),C(:,:,k));
        end
        for k = 1:K
            pk(i,k) = piNormDist(k)/sum(piNormDist(:),1);
        end
    end

    %% M-step
    for k = 1:K
        sum_m = 0;
        sum_C = zeros(2,2);
        for i = 1:N
            sum_m = sum_m + pk(i,k) * X(i,:);
            sum_C = sum_C + pk(i,k) * (X(i,:)-m(k,:))'*(X(i,:)-m(k,:));
        end
        m(k,:) = sum_m./sum(pk(:,k),1);
        C(:,:,k) = sum_C./sum(pk(:,k),1);
        pi(k) = sum(pk(:,k),1)/N; 
    end  
end

%% Calculate averages
pk_ave = zeros(3,K);
n = zeros(3,1);
for i = 1:N
    for l = 1:3      
        if Z(i,l) == 1
            pk_ave(l,:) = pk_ave(l,:) + pk(i,:);
            n(l) = n(l) + 1;
        end
       
    end
end
pk_ave(1,:) = pk_ave(1,:)./n(1);
pk_ave(2,:) = pk_ave(2,:)./n(2);
pk_ave(3,:) = pk_ave(3,:)./n(3);


