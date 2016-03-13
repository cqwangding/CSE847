load('diabetes.mat');
lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
num_lambda = length(lambda);
num_train = length(y_train);
num_test = length(y_test);
MSE_train = zeros(num_lambda,1);
MSE_test = zeros(num_lambda,1);
MSE_validation = zeros(num_lambda,1);

% Calculate MSE using my ridge regression solver
for i=1:length(lambda)
    w = myridge(y_train,x_train,lambda(i));
    MSE_train(i) = mean((y_train-x_train*w).^2);
    MSE_test(i) = mean((y_test-x_test*w).^2);
end

% Set 5 fold validation
idx = crossvalind('Kfold', num_train, 5);
for i=1:num_lambda
    predict=zeros(num_test,1);
    for j=1:5
        idx_train=find(idx~=j);
        idx_test=find(idx==j);
        w = myridge(y_train(idx_train),x_train(idx_train,:),lambda(i));
        MSE_validation(i) = MSE_validation(i) + sum((y_train(idx_test)-x_train(idx_test,:)*w).^2);
    end
end
MSE_validation = MSE_validation/num_train;
% Find the best lambda from 5-fold cross validation procedure
[~,idx_best] = min(MSE_validation);
fprintf('The best lambda is: %f\n',lambda(idx_best));

% Plot MSE
figure;
hold on;
plot(lambda,MSE_train,'o-');
plot(lambda,MSE_test,'x-');
plot(lambda,MSE_validation,'square-');
hold off;
set(gca,'xscale','log');
xlabel('\lambda');
ylabel('MSE');
legend('Training MSE','Testing MSE','5-fold cross validation MSE')
box on;

