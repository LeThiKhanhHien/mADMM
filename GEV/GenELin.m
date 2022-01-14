function [e,feas,t,y] =GenELin(C,B,options) 
% solve max <y,Cy> subject to <y,By>=1 by GenELin proposed in 
% Geetal, "Efficient algorithms for large-scale generalized eigenvector computation
% and canonical correlation analysis",  ICML 2016
% linear equation in this code is solved by Matlab’s backslash 
% output: sequence of objective function e 
%         sequence of feasibility error feas
%         sequence of running time t 
%         solution y
% written by LTK Hien
starttime=tic;
[q,~]=size(B);

%% Parameters 
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'maxiter')
    options.maxiter = inf; 
end
if ~isfield(options,'timemax')
    options.timemax = 100; 
end
if ~isfield(options,'y')
   y = rand(q,1);
else
   y= options.y; 
end


i=1;
y = rand(q,1); 
y = y/norm(y);
y=y/sqrt(sum(y.*(B*y)));
time_err0=tic;
e(i)= sum(y.*(C*y));
feas(i)=abs(sum(y.*(B*y))-1);
time_err=toc(time_err0);
t(i)=toc(starttime) - time_err;

while i <= options.maxiter && t(i) < options.timemax %&& gradnorm(i)>1e-6
    
     %update y
     y=B\(C*y);
     y=y/sqrt(sum(y.*(B*y)));
     % compute relative fitting error and orthogonal error; the computed time is not counted. 
      i=i+1;
      %gradnorm(i)=norm(C*y);
      time_err0=tic;
      e(i)= sum(y.*(C*y));
      feas(i)=abs(sum(y.*(B*y))-1);
      time_err=toc(time_err0)+time_err;
      t(i)=toc(starttime) - time_err;
    
    if mod(i,2)==0
        fprintf('GenELin, iter  %d, objective: %1.2e, feasible gap: %1.2e, run time %1.2e \n',i,e(i), feas(i), t(i));     
    end
       
end
end
