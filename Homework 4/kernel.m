function G = kernel(U,V)
l = 0.1;
G = exp(-(U.^2+V.^2)/(2*l^2));
end

