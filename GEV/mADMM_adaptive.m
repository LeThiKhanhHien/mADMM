function [e,feas,t,y] = mADMM_adaptive(C,B,options) 
% solve min <y,Cy> subject to <y,By>=1 by mADMM;
% subproblem is solved by fminunc with trust region method;
% note that fminunc is very slow when the dimension is high.
%
% output: sequence of objective function e 
%         sequence of feasibility error feas
%         sequence of running time t 
%         solution y
% 
% input:  C, B and options.maxiter (max number of iterations),
%         options.delta (default = 1e-2), options.timemax (max running time),  
%         and options.y (initial point)
%
% written by LTK Hien
% last update January 2022

starttime=tic;
[q,~]=size(B);

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
   y=ones(q,1);
else
   y= options.y; 
end


normC =norm(C); 
Lh=2*normC;

normB=norm(B);
Lxi=2*normB; 
epsilon=1/sqrt(normB)-eps;
lambda_min=min(eig(B));
sigma_lower =lambda_min*epsilon;

if ~isfield(options,'delta')
  delta=1e-2;
else
   delta= options.delta; 
end


sigma=1;

i=1;

w=0; %initial
coeff=1.1;

time_err0=tic;
e(i)= sum(y.*(C*y));
feas(i)=sum(y.*(B*y))-1;
time_err=toc(time_err0);
t(i)=toc(starttime) - time_err;

coef1=12/delta/sigma_lower^2;
My=sqrt((2-epsilon^2*normB)/lambda_min);
Mh=normC*My;

beta_bar=coef1*(Lh^2+2*delta^2+2/3*(2*delta^2*My^2+Mh));
beta=12/delta/sigma^2*(Lh^2+2*delta^2+1/3*Lxi^2*abs(w)^2); 


while i <= options.maxiter && t(i) < options.timemax 
    
     %update y
     y=updatey(B,C,y,beta,w,delta,q);
     
     %update w
     deltaw=beta*(sum(y.*(B*y))-1);
     w= w + deltaw;
     %w_trace=[w_trace,w];
     normy=norm(y);
     if normy<epsilon || normy>My
        beta=beta*coeff;
     end
     
     if abs(deltaw)*norm(2*B*y)<sigma*w 
         sigma=max(sigma/1.1,sigma_lower );
         beta=min(beta_bar,12/delta/sigma^2*(Lh^2+2*delta^2+1/3*Lxi^2*abs(w)^2));
       
     end 
      
     % compute relative fitting error and orthogonal error; the computed time is not counted. 
      i=i+1;
      time_err0=tic;
      e(i)= sum(y.*(C*y));
      feas(i)=abs(sum(y.*(B*y))-1);
      time_err=toc(time_err0) + time_err;
      t(i)=toc(starttime) - time_err;

     
    if mod(i,2)==0
        fprintf('mADMM-adaptive, iter  %d, objective: %1.2e, feasible gap: %1.2e, run time %1.2e \n',i,e(i), feas(i), t(i));     
    end
       
end
end
function y=updatey(B,C,yk,beta,w,delta,q)
% solve unconstrained smooth problem 
% min <y,Cy> + w(<y,By>-1) + beta/2 (<y,By>-1)^2 + delta/2||y-y^k||^2
options2 = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','TolFun',1e-6,'Display','off','MaxIter',20);
y = fminunc(@(y) myfun(y,B,C,yk,beta,w,delta,q),yk,options2);
end
function [Lbeta,grad,hessian]=myfun(y,B,C,yk,beta,w,delta,q)
By=B*y; 
Cy=C*y; 
yBy=sum(y.*(By))-1;
Lbeta= sum(y.*(Cy)) + w*yBy + beta/2*yBy^2+delta/2*norm(y-yk)^2;
grad=2*Cy + 2*w*By  + 2*beta*yBy*By + delta*(y-yk);
hessian=2*C+2*(w+beta*yBy)*B + 4*beta*By*(By')+ delta*eye(q);
end
