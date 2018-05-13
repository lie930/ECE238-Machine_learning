function plot_table( pk,K_max,str )

for K = 2:K_max
    disp(['Plotting table of P(k|i) generated from ' str ' with K = ' num2str(K)]);
    pk(:,1:K,K)
end