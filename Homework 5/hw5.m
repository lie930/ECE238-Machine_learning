%% Homework 5, ECE283, Morten Lie. Collaboration with Sondre Kongsgård, Brage Sæther, Anders Vagle

clear all;
close all;
clc;
addpath('functions');

%% Parameters
N = 200;
d = 100;
u = zeros(d, 6);
sigma2 = 0.01;
K_max = 5;
n_rand_inits = 5;

%% Data generation
for j = 1:6
    u(:,j) = generate_random_vector(d);
    while check_orthogonality(u,j) == 0
        u(:,j) = generate_random_vector(d);
    end
end

[X_d, Z_d] = generate_sample_data(u,sigma2,N);

%% PCA
[U,S,V] = svd(X_d);

% Calculate mean of eigenvalues, and find the number of dominant eigenvalues
eig_mean = eigenvalue_mean(S);
d0 = 0;
for d0 = 0:d
    if S(d0+1,d0+1) < eig_mean
        break
    end
end

S_r = zeros(d0,d0);
for i = 1:d0
    S_r(i,i) = S(i,i);
end
U_r = U(:,1:d0);
V_r = V(:,1:d0);
X_r = U_r*S_r*V_r';
X_d0 = X_r*V_r;

%% d-dimensional k-means
m_opt_d0 = zeros(K_max,size(X_d0,2),K_max);
C_opt_d0 = zeros(N,K_max);
for K = 2:K_max  
    min_sme = inf;
    for i = 1:n_rand_inits
        C_d0 = randi(K,N,1);
        [m_d0,C_d0] = k_means(N,K,C_d0,X_d0);
        sme = SME(m_d0,X_d0,C_d0);
        if sme < min_sme
            m_opt_d0(1:K,:,K) = m_d0;
            C_opt_d0(:,K) = C_d0;
            min_sme = sme;
        end
    end
end

% One-hot encoding
a = zeros(N,K_max,K_max);
for K = 2:K_max
    for i = 1:N     
        a(i,C_opt_d0(i,K),K) = 1;
    end
end

%% Generate empirical probability table for d-dimensional data
pk_kmeans_d = zeros(3,K_max,K_max);
for K = 2:K_max
    for l = 1:3
        for k = 1:K
            num_k_l = 0;
            num_l = 0;
            for i = 1:N
                if Z_d(i,l) == 1
                    num_l = num_l + 1;
                    if a(i,k,K) == 1
                        num_k_l = num_k_l + 1;
                    end
                end
            end
            pk_kmeans_d(l,k,K) = num_k_l/num_l;         
        end
    end
end
plot_table(pk_kmeans_d,K_max,'d-dimensional K-means');

%% Random Projections and Compressed Sensing
