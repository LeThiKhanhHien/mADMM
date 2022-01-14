function obj=eval_h(b,A,x1,x2,x3,q)
% evaluate the value of h
% version that avoids to compute e^t with t being positive
 ai_x1 = x1'*A;
 ai_x2 = x2'*A;
 phix = (ai_x1.^2 + ai_x2 + x3)' ;
 b_phix=b.*phix;
 nn  = b_phix(b_phix>=0);
 neg = b_phix(b_phix<0);
 e_neg=exp(neg);
 obj=1/q* (sum(log(1+exp(-nn))) + sum(log(1+e_neg) -neg) );
end