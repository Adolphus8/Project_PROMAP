function [output] = enhance(data, Ndata, corrmat, pc)
%% The "enhance" function-handle serves to enhance a given input data-set. 
% The description is as follows:
%
% Function input(s):
% - data:    The 1 x dim data array goes here whose rows (first dimension) 
% are the data entries while the columns (second dimension) are the variables;
% - Ndata:   The number of "enhanced" data-set to generate from input data-set;
% - corrmat: The dim x dim correlation matrix;
% - pc:      The scalar value of the percentage perturbation.
%
% Function output(s):
% - output: The Ndata x dim matrix of enhanced data-set;
%
%% The function details:

% Pre-function check:
assert(size(data,2) == size(corrmat,2))

% Find the zero entries in the data:
idx = find(data == 0);

% Construct the covariance matrix:
dim = size(data,2);                       % Define the number of features / dimensionality
mat_nom = zeros(dim,dim);                 % Compute covariance matrix w/o accounting for correlations 
for i = 1:dim
for j = 1:dim
pct = pc/100;                             % Percentage perturbation
nom_val = (pct.^2) .* data(i) .* data(j); % Set the covariance to be 1% of the data value  

if nom_val == 0
mat_nom(i,j) = pct.^2;
else
mat_nom(i,j) = nom_val;
end

end
end
covmat = mat_nom .* corrmat; % Define the covariance matrix accounting for correlations

% Generate the sythetic data from the input data:
output = mvnrnd(data, covmat, Ndata); 

% Ensure the zero entries are set as zeros in the final synthetic data-set:
for j = 1:Ndata
output(j, idx) = 0;
end

%% End of function
end

