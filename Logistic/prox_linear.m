function [e,t,x1,x2,x3] =prox_linear(A,b,lambda,options) 
% solve  min g(x) + h(y) s.t \phi(x) - y = 0
% where h(y)=1/q \sum log(1 + e^{-b_i y_i})
%       g(x) = \lambda_1 ||x_1||_1 + \lambda_2 ||x_2||_1
%       \phi_i(x)= <a_i,x_1>^2 + <a_i,x_2> + x_3
% by the prox-linear method proposed in Drusvyatskiy & Paquette, "Efficiency of 
% minimizing compositions of convex functions and smooth maps", MathProg 2019.
% Subproblem is approximately solved by APG.
%
% input   A=[a_i] \in R^{d \times q}, b=(b_i)\in R^q, b_i \in {-1,1}, and \lambda
%         options.maxiter (max number of iterations), 
%         options.timemax (max running time),  
%         and options.x1, options.x2, options.x3, options.y (initial point,
%                         by default use rand for x and zeros for y)
%         options.beta (penalty parameter - use  beta=10*Lh by default)
%
% output: sequence of objective function e 
%         sequence of running time t 
%         solution (x1,x2,x3)
%
% written by LTK Hien, Jan 2022
% 
starttime=tic;
[d,q]=size(A); % sample size = q, 

 lambda1=lambda(1);
 lambda2=lambda(2);
%% Parameters of NMF algorithm
if nargin < 4
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'x1')
   x1 = rand(d,1);
else
   x1= options.x1; 
end

if ~isfield(options,'x2')
   x2 = rand(d,1);
else
   x2= options.x2; 
end

if ~isfield(options,'x3')
   x3 =  rand(1);
else
   x3= options.x3; 
end


if ~isfield(options,'maxiter')
    options.maxiter = inf; 
end
if ~isfield(options,'timemax')
    options.timemax = 10; 
end

  
i=1;

time_err0=tic; % exclude computing time of the objective 
e(i)= lambda1*sum(abs(x1)) + lambda2*sum(abs(x2)) + eval_h(b,A,x1,x2,x3,q);
time_err=toc(time_err0);
t(i)=toc(starttime) - time_err;


normA=sqrt(sum(A.^2));
Ltheta=sum(normA);
L=Ltheta/(4*q); % 1/tau



while i <= options.maxiter && t(i) < options.timemax
    % update x
     [x1,x2,x3]=APG(A,b,x1,x2,x3,L,d,lambda1,lambda2,q);
     
    % compute objective function, the computed time is not counted. 
      i=i+1;
    
      time_err0=tic;
      e(i)= lambda1*sum(abs(x1))+ lambda2*sum(abs(x2)) + eval_h(b,A,x1,x2,x3,q);
      time_err=toc(time_err0) + time_err;
      t(i)=toc(starttime) - time_err;
      
    
    if mod(i,2)==0
        fprintf('prox_linear,iter  %d, fitting error: %1.2e, running time: %1.2e  \n',i,e(i),t(i));     
    end
end
end


function [x1,x2,x3]=APG(A,b,x1k,x2k,x3k,L,d,lambda1,lambda2,q)
% take initial point to be x1k,x2k,x3k
x1=x1k;
x2=x2k;
x3=x3k;
ai_x1 = x1k'*A; 
ai_x2 = x2k'*A; 
bt=b';
phixk = bt.*( ai_x1.^2 + ai_x2 + x3); 
nablac_i = 2*bt.*ai_x1 ; %  coefficient of \nabla_{x_1} \tilde \phi(xk)


stepsize=1e-2; 
% note: we tried stepsize = 1/L_constant but it does not work well, 
% L_constant is the Lipschitz-smooth constant of the subproblem, which can
% be estimated by       L_constant = L_h*||\nabla \phi(x^k)||^2 + L; 
%                       stepsize = 1/L_constant;  
% The value 1e-2 seems to be the best choice for the stepsize
% 
% decrease the stepsize if needed
epsilon=1e-3;
dist=1;
i=1;
x1_prev=x1;
x2_prev=x2;
x3_prev=x3;
t_prev=1;   

while dist>epsilon && i< 1e3 % max number of iteration is 1000. 
    
      t=1/2*(1+sqrt(1+4*t_prev^2)); 
      coef_extra=(1-t_prev)/t; 
      t_prev=t;
      
      x1_extra= x1 + coef_extra* (x1-x1_prev);
      x2_extra= x2 + coef_extra* (x2-x2_prev);
      x3_extra= x3 + coef_extra* (x3-x3_prev);
      
      % find gradient 
      ai_x1xk=(x1_extra-x1k)'*A; 
      nablac_xxk=nablac_i.* ai_x1xk; % \nabla \tilde\phi(x^k)^T (x-x^k)
      

      temp=(phixk + nablac_xxk);
      signy0=sign(temp);
      signy=signy0;
      signy0(signy0<0)=0;
      denominator=1+exp(-temp.*signy);
      nominator=exp(-temp.*signy0);
      
           
      coeff_x2=-1/q* nominator./denominator;
      coeff = (coeff_x2.*nablac_i); 
      
      grad_x1 = sum( repmat(coeff,d,1).*A,2) + L*(x1_extra-x1k); 
      grad_x2 = sum( repmat(bt.*coeff_x2,d,1).*A,2); 
      grad_x3 = sum(bt.*coeff_x2);
      
      x1_prev=x1;
      x2_prev=x2;
      x3_prev=x3;

      %update x1
      to_find_prox= x1_extra-stepsize*grad_x1;
      x1= sign(to_find_prox).* max(abs(to_find_prox)-lambda1*stepsize,eps);
      %update x2
      to_find_prox= x2_extra-stepsize*grad_x2;
      x2= sign(to_find_prox).* max(abs(to_find_prox)-lambda2*stepsize,eps);
      
      
      %update x3
      x3= x3_extra-stepsize*grad_x3;
      i=i+1;
      
      dist= norm(x1_prev-x1)+ norm(x2_prev-x2)+ norm(x3_prev-x3);

     
end

end