function plot_mean_comp(pk,K,u,m_kmeans)
d0 = size(m_kmeans,2);

% Project the u vectors that contribute to the expected value of each component down to d0 dimensions
a = zeros(size(pk,1),K);
E_u = [u(1,:)' 2*u(4,:)' sqrt(2)*u(6,:)'];

for k = 1:K
    E_empr = zeros(size(u,2),1);
    a(:,k) = pk(:,k,K);
    for c=1:3
        E_empr = E_empr + a(c,k)*E_u(:,c);
    end
    E_empr = E_empr./sum(a(:,k));
    
    figure();
    pcolor(1:d0,0:K,[E_empr';m_kmeans(k,:);zeros(1,d0)]);
    title(['Component ' num2str(k) ', ' num2str(d0) ' dimensions']);
    ylabel('Expected mean            Computed mean');
end
