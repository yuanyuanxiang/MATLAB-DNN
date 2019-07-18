function m = reLU(m)
% 激活函数：reLU,S函数或其他，视具体情况而定。
% y = f(x): x-Input

%i=m<=0; m(i)=0; %reLU
m = 1./(1+exp(-m)); %Sigmoid
%a = exp(m); b = exp(-m); m = (a-b)./(a+b); %tanh
end
