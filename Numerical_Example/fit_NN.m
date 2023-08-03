function [net,tr] = fit_NN(inputs, targets, model_no) 
%% This is the function used to solve an Input-Output Fitting problem with a Neural Network
% 
% Usage:
% [net, tr] = fit_NN(inputs, targets) 
%
% Inputs:
% inputs:  The Ndata x Dim_input matrix of model inputs;
% targets: The Ndata x Dim_output matrix of model outputs;
% model_no: Scalar value of the ANN architecture to use;
%
% Output:
% net: The trained ANN model function;
% tr:  The structure of the ANN training statistics;   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define the function:

% Create a series of different Fitting Network:
% Advice - Do not run more than 3 hidden layers
if model_no == 1
hidden_nodes_vec = [18];          % ANN architecture type 1
elseif model_no == 2
hidden_nodes_vec = [18, 9];       % ANN architecture type 2
elseif model_no == 3
hidden_nodes_vec = [27, 18, 9];   % ANN architecture type 3
elseif model_no == 4
hidden_nodes_vec = [64, 32, 16];  % ANN architecture type 4
elseif model_no == 5
hidden_nodes_vec = [32];          % ANN architecture type 5
elseif model_no == 6
hidden_nodes_vec = [64];          % ANN architecture type 6
elseif model_no == 7
hidden_nodes_vec = [64, 32, 8];   % ANN architecture type 7
elseif model_no == 8
hidden_nodes_vec = [400, 350, 300, 250, 200, 150, 100, 50];  % ANN architecture type 8
elseif model_no == 9
hidden_nodes_vec = [2];  % ANN architecture type 9
elseif model_no == 10
hidden_nodes_vec = [4];  % ANN architecture type 10
elseif model_no == 11
hidden_nodes_vec = [6];  % ANN architecture type 11
elseif model_no == 12
hidden_nodes_vec = [8];  % ANN architecture type 12
elseif model_no == 13
hidden_nodes_vec = [1];  % ANN architecture type 13
end

% Specify the type of ANN training and epochs:
net = feedforwardnet(hidden_nodes_vec); % Training type: Feed-forward net
net.trainParam.epochs = 1000;

% Set activation function for each hidden layer as RELU :
for i = 1:length(hidden_nodes_vec)
%net.layers{i}.transferFcn = 'poslin'; % RELU (Rectified Linear Unit activation)
net.layers{i}.transferFcn = 'logsig'; % Logistic Sigmoid function
%net.layers{i}.transferFcn = 'purelin'; % Linear function
end

% Set up Division of Data for Training and Testing:
net.divideParam.trainRatio = 0.6;
net.divideParam.testRatio = 0.2;
net.divideParam.valRatio = 0.2;

% Train the Network:
[net,tr] = train(net,inputs',targets');

end