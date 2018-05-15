clear all;
close all;
clc;

N = 50;
%% Class definitions
% Class 0 definition
theta1 = 0;
m = [0 0]';
lambda_1 = 2;
lambda_2 = 1;
u_1 = [cos(theta1) sin(theta1)]';
u_2 = [-sin(theta1) cos(theta1)]';
C1 = [u_1 u_2]*diag([lambda_1,lambda_2])*inv([u_1 u_2]);
C1_pts = mvnrnd(m,C1,N/2);

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
gm = gmdistribution([m_a';m_b'],C2,[pi_a pi_b]);
C2_pts = random(gm,N/2);
X = [C1_pts;C2_pts];
Y = [ones(N/2,1);-1*ones(N/2,1)];
sample_min = [min(X(1,:)), min(X(2,:))];
sample_max = [max(X(1,:)), max(X(2,:))];



%% SVM, using Matlab's built in function fitcsvm

[x1Grid,x2Grid] = meshgrid(sample_min(1):0.1:sample_max(1),sample_min(2):0.1:sample_max(2));
xGrid = [x1Grid(:),x2Grid(:)];
SVM_model = fitcsvm(X,Y,'KernelFunction','gaussian');
[~,scores] = predict(SVM_model,xGrid);

figure
hold on;
scatter(C1_pts(:,1), C1_pts(:,2),'r');
scatter(C2_pts(:,1), C2_pts(:,2),'b');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
hold off;
title('Datapoints');
legend('Dataset 1','Dataset 2');

