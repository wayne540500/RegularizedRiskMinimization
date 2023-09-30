% loading dataset
clc;
clear;
load problem2.mat;


% split data into -> training set and testing set

% xt = vector of input scalars for training
xt = x(1:round(length(x)/2),:);
% xT = vector of output scalars for training
xT = x(round(length(x)/2)+1:end,:);
% yt = vector of input scalars for testing
yt = y(1:round(length(x)/2),:);
% yT = vector of output scalars for testing
yT = y(round(length(x)/2)+1:end,:);

% Due to range for this data set would be from λ = 0 to λ = 1000
% create a array of all zeros
test_err = zeros(1000,1);
train_err = zeros(1000,1);
test_err_notreg = zeros(1000,1);
train_err_notreg = zeros(1000,1);

% fit the model from many Lambda values

i = 1;
for lamb = 0:1000
[err,err_notreg,model,errT,errT_notreg] = reg(xt,yt,lamb,xT,yT);
test_err(i) = errT;
train_err(i) = err;
test_err_notreg(i) = errT_notreg;
train_err_notreg(i) = err_notreg;
i = i + 1;
end

figure(1)
plot(test_err(1:1000),'g');
hold on
plot(train_err(1:1000),'r');
xlabel('Lambda');
ylabel('Error');
legend('Test Error' , 'Train Error');

figure(2)
plot(test_err_notreg(1:1000),'g');
hold on
plot(train_err_notreg(1:1000),'r');
xlabel('Lambda');
ylabel('Error');
legend('Test Error' , 'Train Error');

%Regularized risk minimization Function

function [err,err_notreg,model,errT,errT_notreg] = reg(x,y,lambda,xT,yT)
[n,m] = size(x);
model = inv(x'*x + lambda*eye(m))*(x'*y);
err = (1/(2*length(x)))*sum((y-x*model).^2) + (lambda/(2*length(x))) * (model'*model);
err_notreg= (1/(2*length(x)))*sum((y-x*model).^2) ;
if (nargin==5)
errT = (1/(2*length(xT)))*sum((yT-xT*model).^2) + (lambda/(2*length(xT))) *(model'*model);
errT_notreg = (1/(2*length(xT)))*sum((yT-xT*model).^2) ;
end
end
