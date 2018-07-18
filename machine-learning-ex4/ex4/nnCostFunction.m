function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

a1 = [ones(m, 1) X];
z2 = (a1 * Theta1');
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = sigmoid(a2 * Theta2');

% disp(size(X));
% disp(size(a1));

% disp(size(z2));
% disp(size(a2));

% disp(size(Theta2));
% disp(size(z3));

% disp(y);

yLog = 0;
oneMinusYLog = 0;
for l = 1:num_labels
  k = y == l;
  yLog = k .* log(z3(:,l));
  oneMinusYLog = (1 - k) .* log(1 - z3(:,l));
  Jl = (1/m * sum(-yLog - oneMinusYLog));
  J = J + Jl;
end

% Regularized Cost Function
J = J + (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

y_numlabels = eye(num_labels)(y,:);
% disp(size(y_numlabels));
a3 = z3;
d3 = a3 - y_numlabels;
% disp(size(d3));
sigGra = sigmoidGradient(z2);
sigGra = [ones(m,1) sigGra];
% disp(size(sigGra));

% disp(size(Theta2));
% disp(size(d3));
% disp(size((d3 * Theta2)));
d2 = (d3 * Theta2) .* sigGra;
% disp(size(d2));
d2 = d2(:, 2:end);
% disp(size(d2));

delta1 = d2' * a1;
delta2 = d3' * a2;

Theta1_grad = Theta1_grad + (1/m) * delta1;
Theta2_grad = Theta2_grad + (1/m) * delta2;


% grad = my_grad + ((lambda/m) .* theta);

% function [J grad] = nnCostFunction(nn_params, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, ...
%                                    X, y, lambda)



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
