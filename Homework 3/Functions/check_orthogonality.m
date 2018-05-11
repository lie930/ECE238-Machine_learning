function [ orthogonal ] = check_orthogonality( u,j )
u_new = u(:,j);
max_sum = 0;
for x = 1:j-1
    sum = 0;
    for y = 1:size(u,1)
        sum = sum + abs(u(y,x)*u_new(y));
    end
    if sum > max_sum
        max_sum = sum;
    end
end

if max_sum/size(u,1) < 0.2 || j == 1
    orthogonal = 1;
else
    orthogonal = 0;
end
end