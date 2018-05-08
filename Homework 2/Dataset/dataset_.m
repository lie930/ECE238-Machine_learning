clc;
clear all;
close all;

N = 1000;

%% Class definitions
% Class 0 definition
theta1 = 0;
m = [0 0]';
lambda_1 = 2;
lambda_2 = 1;
u_1 = [cos(theta1) sin(theta1)]';
u_2 = [-sin(theta1) cos(theta1)]';
C1 = [u_1 u_2]*diag([lambda_1,lambda_2])*inv([u_1 u_2]);
C1_pts = mvnrnd(m,C1,N);
C1_pts = [C1_pts zeros(N,1)];

% Class 1 definition
theta2_a = -3*pi/4;
m_a = [-2 1]';
pi_a = 1/3;
lambda_a1 = 2;
lambda_a2 = 1/4;
u1_a = [cos(theta2_a) sin(theta2_a)]';
u2_a = [-sin(theta2_a) cos(theta2_a)]';
C_a = [u1_a u2_a]*diag([lambda_a1,lambda_a2])*inv([u1_a u2_a]);

theta2_b = pi/4;
pi_b = 2/3;
m_b = [3 2]';
lambda_b1 = 3;
lambda_b2 = 1;
u1_b = [cos(theta2_b) sin(theta2_b)]';
u2_b = [-sin(theta2_b) cos(theta2_b)]';
C_b = [u1_b u2_b]*diag([lambda_b1,lambda_b2])*inv([u1_b u2_b]);
C2(:,:,1) = C_a;
C2(:,:,2) = C_b;
gm = gmdistribution(2*[m_a';m_b'],C2,[pi_a pi_b]);
C2_pts = random(gm,N);
C2_pts = [C2_pts ones(N,1)];

%% Partition data into 3 parts, 70:20:10
data_11 = C1_pts(1:0.7*N,:);
data_12 = C1_pts(1:0.2*N,:);
data_13 = C1_pts(1:0.1*N,:);

data_21 = C2_pts(1:0.7*N,:);
data_22 = C2_pts(1:0.2*N,:);
data_23 = C2_pts(1:0.1*N,:);

dataset_1 = [data_11;data_21];
dataset_2 = [data_12;data_22];
dataset_3 = [data_13;data_23];

save('train_dataset_2mean.txt','dataset_1','-ascii');
save('test_dataset_2mean.txt','dataset_2','-ascii');
save('validation_dataset_2mean.txt','dataset_3','-ascii');