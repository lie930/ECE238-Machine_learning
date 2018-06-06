function [ eig_mean ] = eigenvalue_mean( S )
eig_mean = 0;
for i = 1:min(size(S,1),size(S,2))
    eig_mean = eig_mean + S(i,i);
end
eig_mean = eig_mean/min(size(S,1),size(S,2));
end

