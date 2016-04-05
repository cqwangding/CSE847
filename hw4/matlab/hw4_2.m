% load alzheimers data
load('alzheimers/ad_data.mat');

X_train = [X_train, ones(size(X_train,1),1)];
X_test = [X_test, ones(size(X_test,1),1)];

% L1 parameters
par  = [0 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

accuracy = zeros(length(par),1);
AUC = zeros(length(par),1);
num_feature = zeros(length(par),1);
for i=1:length(par)
    % training
    [w c] = logistic_l1_train(X_train, y_train, par(i));
    % testing
    y = logsig(X_test*w);
    [X,Y,T,AUC(i)] = perfcurve(y_test,y,1);
    num_feature(i) = sum(w~=0);
end

% plot AUC, number of features selected
figure;
plot(par, AUC,'x-');
xlabel('L1 parameter');
ylabel('AUC');
box on;

figure;
plot(par, num_feature,'x-');
xlabel('L1 parameter');
ylabel('Num of features selected');
box on;
