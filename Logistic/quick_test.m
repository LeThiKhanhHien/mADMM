% logistic regression with nonlinear classifier
close all; clear all; clc

% generate data 
 d=1000; q=100; 
 A=rand(d,q);
 normA=sqrt(sum(A.^2)); %scale A: to avoid numerical error we should scale A such that each of its 
                        % column has norm equal to 1 for the input
 A=A./repmat(normA,d,1);

 b=randsample([-1, 1],q,true);
 b=b'; 

 %initial 
 x1 = rand(d,1); x1=x1/norm(x1);
 x2 = rand(d,1); x2=x2/norm(x2);
 x3=rand(1);
 
 options.x1=x1;
 options.x2=x2;
 options.x3=x3;
 options.y=zeros(q,1);
 
 % set running time
 options.timemax=10;
 
 % choose penalty parameter
 options.beta = 2.5/q;
 
 % choose regularizer parameter
 lambda=[0.001,0.1]; 
  
   % run algorithm
 [e,t,x1,x2,x3] = mADMM(A,b,lambda,options);  % run mADMM 
 [e_pl,t_pl,x1_pl,x2_pl,x3_pl] = prox_linear(A,b,lambda,options); % run prox-linear
 
 % graph 
 figure;
 set(0, 'DefaultAxesFontSize', 18);
 set(0, 'DefaultLineLineWidth', 2);

 semilogy(t,e,'b','LineWidth',3);hold on; 
 semilogy(t_pl,e_pl,'r--','LineWidth',3);hold on; 

 ylabel('Fitting error');
 xlabel('Time'); 
 legend('mADMM','prox-linear');
 