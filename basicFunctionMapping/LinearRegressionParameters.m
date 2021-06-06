function [theta] = LinearRegressionParameters(Q, y)


theta = zeros(size(Q, 2), 1);
theta = pinv(Q' * Q) * Q' * y;
end