function [ X,Z ] = generate_sample_data(u,sigma2,N)
d = size(u,1);
X = zeros(N,d);
Z = zeros(N,3);
noise = zeros(d,1);
for i = 1:N
    Z1 = normrnd(0,1);
    Z2 = normrnd(0,1);
    noise_matrix = normrnd(0,sigma2*eye(d));
    for j = 1:d
        noise(j) = noise_matrix(j,j);
    end
    
    comp_select = rand(1);
    if comp_select < 1/3
        X(i,:) = (u(:,1) + Z1*u(:,2) + Z2*u(:,3) + noise)';
        Z(i,:) = [1 0 0];
    elseif comp_select < 2/3
        X(i,:) = (2*u(:,4) + sqrt(2)*Z1*u(:,5) + Z2*u(:,6) + noise)';
        Z(i,:) = [0 1 0];
    else
        X(i,:) = (sqrt(2)*u(:,6) + Z1*(u(:,1)+u(:,2)) + 1/sqrt(2)*Z2*u(:,5) + noise)';
        Z(i,:) = [0 0 1];
    end
end

end

