function w = myridge( y, X, lambda )
% ridge regression solver
num = size(X,2);
w = inv(X'*X+lambda*eye(num))*X'*y;
end

