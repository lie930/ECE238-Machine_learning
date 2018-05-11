function [ u ] = generate_random_vector( d )
u = zeros(d,1);
P_1 = 2/3;
P_2 = 1/6;
for i = 1:d
    a = rand(1);
    if a < P_1
        u(i) = 0;
    elseif a < P_1 + P_2
        u(i) = 1;
    else
        u(i) = -1;
    end
end