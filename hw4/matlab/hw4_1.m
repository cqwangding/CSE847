% read spam_email data
data =  dlmread('spam_email/data.txt');
labels = dlmread('spam_email/labels.txt');
data = [data, ones(size(data,1),1)];

% create a separate test data set
test = data(2001:4601,:);
test_labels = labels(2001:4601,:);

% choose first n rows of the training data
n = [200 500 800 1000 1500 2000];
accuracy = zeros(length(n),1);
for i=1:length(n)
    train = data(1:n(i),:);
    train_labels = labels(1:n(i),:);
    % training
    w = logistic_train(train, train_labels);
    % testing
    y = logsig(data*w);
    y(y>=0.5) = 1;
    y(y<0.5) = 0;
    accuracy(i) = sum(y==labels)/length(labels);
end

% plot accuracy
figure;
plot(n, accuracy,'x-');
xlabel('Training with n rows');
ylabel('Accuracy on test data');
box on;
