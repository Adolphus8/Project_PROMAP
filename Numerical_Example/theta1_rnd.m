function [output] = theta1_rnd(N)
%% Function-handle: Random number generator for Theta 1:

% theta_1 ~ [300, 500] U [600, 700] N; 
%% Define the parameter:
Nsamps = N;          % No. of samples
output = zeros(N,1); % Sample output vector
idx = randi([1,2],Nsamps, 1);

for i = 1:Nsamps
kdx = idx(i);

if kdx == 1
output(i) = unifrnd(300, 500, 1, 1);
elseif kdx == 2
output(i) = unifrnd(600, 700, 1, 1);  
end
    
end

%% 
end

