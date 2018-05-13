function [ m,C ] = k_means( N,K,C,X )
convergence = 0;
m = zeros(K,size(X,2));
C_new = zeros(N,1);
while convergence == 0
    % Update mean
    for k = 1:K
        norm = 0;
        for i = 1:N
            if C(i) == k
                norm = norm + 1;
                m(k,:) = m(k,:) + X(i,:);
            end
        end
        m(k,:) = m(k,:)/norm;
    end

    % Update component vector C
    for i = 1:N
        min_dist = inf;
        for k = 1:K
            dist = sqrt((X(i,1)-m(k,1))^2 + (X(i,2)-m(k,2))^2);
            if dist < min_dist
                C_new(i) = k;
                min_dist = dist;
            end
        end
    end

    % Check for convergence
    if sum(abs(C_new - C),1) == 0
        convergence = 1;
    end
    C = C_new;
end