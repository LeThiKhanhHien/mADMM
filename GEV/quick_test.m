close all; clear all; clc
 
fprintf('quick test\n');
% generate data   
q = 500;
C = randn(q);
C = (C+C')/2;
C=C/norm(C);
  
p=0.001;
v = (1:q).^(-p);
v = diag(v);
s = orth(randn(q));
B=   s*v*s';

options.timemax=10;

%% minimum generalized eigenvalue problem

fprintf(' min GEV problem \n');
% find optimal objective. For larger dimension, use manopt to get the optimal value faster.  
opt_minGEV=min(eig(B^(-1/2)*C*B^(-1/2))); 
fprintf('run GenELin\n');
[e_elin,feas_elin,t_elin,y_elin] = GenELin(-C,B,options); 
fprintf('run mADMM\n');
[e,feas,t,y] =  mADMM_adaptive(C,B,options); 


%% maximum generalized eigenvalue problem
fprintf(' max GEV problem\n');
% find optimal objective. For larger dimension, use manopt to get the optimal value faster.  
opt_maxGEV=max(eig(B^(-1/2)*C*B^(-1/2))); 
% run GenELin
fprintf('run GenELin\n');
[e_elin2,feas_elin2,t_elin2,y_elin2] = GenELin(C,B,options); 
% run mADMM
fprintf('run mADMM\n');
[e2,feas2,t2,y2] =  mADMM_adaptive(-C,B,options);

fprintf('Results: \n');
fprintf('****Min GEV problem:**** \n');
fprintf('Optimality gap\n');
fprintf('mADMM: %1.4e,    GenELin: %1.4e\n',abs(e(end)-opt_minGEV),abs(-e_elin(end)-opt_minGEV));
fprintf('Feasibility error\n');
fprintf('mADMM: %1.4e,    GenELin: %1.4e\n\n',feas(end),feas_elin(end));

fprintf('****Max GEV problem:**** \n');
fprintf('Optimality gap\n');
fprintf('mADMM: %1.4e,   GenELin: %1.4e\n',abs(e2(end)+opt_maxGEV), abs(e_elin2(end)-opt_maxGEV));
fprintf('Feasibility error\n');
fprintf('mADMM: %1.4e,   GenELin: %1.4e\n\n',feas2(end),feas_elin2(end));
