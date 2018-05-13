function  plotData(X,Z,m_1,m_2,m_3,K_max,C_opt,m_opt,C_EM,m_EM,N)
h = figure(1);
set(h,'position',[0,0,1000,1000]);
subplot(2,3,1);
hold on;
for i = 1:N
    if Z(i,1) == 1
         scatter(X(i,1),X(i,2),'r')
    elseif Z(i,2) == 1
        scatter(X(i,1),X(i,2),'g');
    elseif Z(i,3) == 1
        scatter(X(i,1),X(i,2),'b');         
    end
end
scatter(m_1(1), m_1(2), 'r','filled');
scatter(m_2(1), m_2(2), 'g','filled');
scatter(m_3(1), m_3(2), 'b','filled');
title('Original components');
hold off;

colors = 'rbkmc'; 
x1 = min(X(:,1)):0.05:max(X(:,1));
x2 = min(X(:,2)):0.05:max(X(:,2));
[X1,X2] = meshgrid(x1,x2);

for K = 2:K_max
    C = C_opt(:,K);
    m = m_opt(:,:,K);
    subplot(2,3,K+1);
    hold on;
    for i = 1:N
        for k = 1:K
            if C(i) == k
                scatter(X(i,1),X(i,2),colors(k));   
            end
        end
    end
    for k = 1:K
        scatter(m(k,1), m(k,2),'filled',colors(k));
        F = mvnpdf([X1(:) X2(:)],m_EM(k,:,K),C_EM(:,:,k,K));
        F = reshape(F,length(x2),length(x1));
        contour(x1,x2,F,3,colors(k),'LineWidth',1.5);
        scatter(m_EM(k,1,K),m_EM(k,2,K),'x',colors(k),'LineWidth',1.5);
    end
    title(['K = ' num2str(K)]);
    hold off;
end
end

