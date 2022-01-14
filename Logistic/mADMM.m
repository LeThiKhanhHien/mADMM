function [e,t,x1,x2,x3] = mADMM(A,b,lambda,options) 
% solve  min g(x) + h(y) s.t \phi(x) - y = 0
% where h(y)=1/q \sum log(1 + e^{-b_i y_i})
%       g(x) = \lambda_1 ||x_1||_1 + \lambda_2 ||x_2||_1
%       \phi_i(x)= <a_i,x_1>^2 + <a_i,x_2> + x_3
% 
% input   A=[a_i] \in R^{d \times q}, b=(b_i)\in R^q, b_i \in {-1,1}, and \lambda
%         options.maxiter (max number of iterations), 
%         options.timemax (max running time),  
%         and options.x1, options.x2, options.x3, options.y (initial point,
%                         by default use rand for x and zeros for y)
%         options.beta (penalty parameter - use  beta=10*Lh by default)
% 
% to avoid numerical error we should scale A such that each of its 
% column has norm equal to 1 for the input
%
% output: sequence of objective function e 
%         sequence of running time t 
%         solution (x1,x2,x3)
%         
%
% written by LTK Hien, Jan 2022
%
starttime=tic;
[d,q]=size(A); % sample size = q, variable dimension = d
Lh=1/4/q;
lambda1=lambda(1);
lambda2=lambda(2);
%% Parameters 
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
   x3 = rand(1);
else
   x3= options.x3; 
end

if ~isfield(options,'y')
   y = zeros(q,1);
else
   y= options.y; 
end

if ~isfield(options,'beta')
   beta=10*Lh;
else
   beta= options.beta; 
end

if ~isfield(options,'maxiter')
    options.maxiter = inf; 
end
if ~isfield(options,'timemax')
    options.timemax = 10; 
end

eps=1e-20; % considered to be 0

% initial 
w=zeros(q,1);
i=1;
time_err0=tic; % computing time of the objective is excluded
e(i)= lambda1*sum(abs(x1))+  lambda2*sum(abs(x2))+ eval_h(b,A,x1,x2,x3,q);
time_err=toc(time_err0);
t(i)=toc(starttime) - time_err;

normA2=(sum(A.^2))'; % a column containing the norm square of each column
l2=beta*sum(normA2);
l3=beta*q;  
beta_Lh= 1/(beta+Lh);

% start 
while i <= options.maxiter && t(i) < options.timemax
  
    % update x1
       ai_x1 = x1'*A;
       ai_x2 = x2'*A; 
       
       yt=y';
       
       phix_yi = ai_x1.^2 + ai_x2 + x3-yt; % phi_i(x) - y_i 
       phix_yi_wi = beta*phix_yi + w'; 
       
       coeff=phix_yi_wi.*ai_x1; 
       grad_varphi = 2* repmat(coeff,d,1).*A; % grad of varphi
       
       gradh = (norm(x1)^2+1)*x1; % grad of h 
        
       max1=abs(w-beta*y)+beta*abs(ai_x2'+x3); 
       to_find_max=[max1 3*beta*normA2];
       
       l1 = 2*normA2.* (max(to_find_max,[],2)); 
       l1=sum(l1); % relative smooth constant 
       
       c_tild=sum(grad_varphi,2) - l1*gradh;
       
       if c_tild ~= 0 
           c_tild_lambda=abs(c_tild)-lambda1;
           nn_c_tild =  max(c_tild_lambda,eps); % eps is considered to be 0 to avoid numerical errors
           c=norm( nn_c_tild );
       else
           c=-lambda1;
       end 
       temp1 = c/2/l1; 
       temp2 = sqrt(1/27+temp1^2);
       s1=nthroot(temp1 + temp2,3);
       s2=nthroot(temp1 - temp2,3);
       
      if c_tild ~= 0 
       Tc=-nn_c_tild.*sign(c_tild); 
       x1=(s1+s2)*Tc/norm(Tc);
      else 
       x1=(s1+s2)*[zeros(d-1,1);1];
      end
       
     
    
    % update x2
       ai_x1=x1'*A;
       ai_x1_2=ai_x1.^2; 
       
       phix_yi_wi=  beta*( ai_x1_2+ ai_x2 + x3-yt) + w';   % x1 was updated, recalculate phix_yi_wi
       grad_x2=repmat(phix_yi_wi,d,1).*A; 
       to_find_prox= x2-1/l2*sum(grad_x2,2);
       x2= sign(to_find_prox).* max(abs(to_find_prox)-lambda2/l2,eps);
       
    % update x3 
      ai_x2 = x2'*A; 
      phi_nox3= ai_x1_2 + ai_x2; 
      phix_yi_wi=  beta*(phi_nox3+ x3-yt) + w';
      grad_x3=sum(phix_yi_wi);
      x3= x3-1/l3*grad_x3;
      
    % update y
      by=b.*y;
      signy0=sign(by);
      signy=signy0;
      signy0(signy0<0)=0;
      
      denominator=1+exp(-by.*signy); % avoid computing e^t with positive t 
      nominator=exp(-by.*signy0);
      gradh_y=-1/q*(b.*nominator)./denominator; 
      
      beta_phix=  beta*( phi_nox3 + x3)';
      phix_wi= beta_phix + w;
      y=beta_Lh*(Lh*y-gradh_y + phix_wi);
      
    % update w
      w= w + beta_phix-beta*y;
      
    % compute fitting error; the computed time is not counted. 
      i=i+1;
    
      time_err0=tic;
      e(i)= lambda1*sum(abs(x1))+ lambda2*sum(abs(x2)) + eval_h(b,A,x1,x2,x3,q);
      time_err=toc(time_err0)+time_err;
      t(i)=toc(starttime) - time_err;

    
    if mod(i,5)==0
        fprintf('mADMM,iter  %d, fitting error: %1.2e, running time: %1.2e  \n',i,e(i),t(i));     
    end
       
end

end
