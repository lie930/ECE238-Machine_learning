function [ sme ] = SME( m,X,C )
assert(size(X,1) == size(C,1));
N = size(X,1);
K = size(m,1);
smes = zeros(K,1);
for i = 1:N
    k = C(i);
    sum_of_squares = 0;
    for j = 1:size(X,2)
        sum_of_squares = sum_of_squares + (m(k,j)-X(i,j))^2;
    end
    smes(k) = smes(k) + sqrt(sum_of_squares);
end
sme = sum(smes,1)/N;
end