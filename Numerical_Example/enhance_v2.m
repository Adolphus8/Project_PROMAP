function [output] = enhance_v2(data, Ndata, corrmat, pc, cse)
%% The "enhance" function-handle serves to enhance a given input data-set. 
% NOTE: This is to be used for the Test case only!
%
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
covmat = mat_nom .* corrmat; % Define the full covariance matrix accounting for correlations

%% Consider the different conditional cases:

if cse == 1 % keep the first variable fixed
    
diagonal_1 = covmat(2,2) - ((covmat(2,1)./covmat(1,1))*covmat(1,2));
diagonal_2 = covmat(3,3) - ((covmat(3,1)./covmat(1,1))*covmat(1,3));
covmat_red = [diagonal_1, 0.9883*sqrt(diagonal_1.*diagonal_2); 0.9883*sqrt(diagonal_1.*diagonal_2), diagonal_2];
output_nom = mvnrnd(data(1,[2,3]), covmat_red, Ndata); % Generate the sythetic data from the input data
output = [ones(Ndata,1).*data(1,1), output_nom];

elseif cse == 2 % keep the second variable fixed

diagonal_1 = covmat(1,1) - ((covmat(1,2)./covmat(2,2))*covmat(2,1));
diagonal_2 = covmat(3,3) - ((covmat(3,2)./covmat(2,2))*covmat(2,3));
covmat_red = [diagonal_1, 0.9930*sqrt(diagonal_1.*diagonal_2); 0.9930*sqrt(diagonal_1.*diagonal_2), diagonal_2];
output_nom = mvnrnd(data(1,[1,3]), covmat_red, Ndata); % Generate the sythetic data from the input data
output = [output_nom(:,1), ones(Ndata,1).*data(1,2), output_nom(:,2)];

elseif cse == 3 % keep the third variable fixed
 
diagonal_1 = covmat(1,1) - ((covmat(1,3)./covmat(3,3))*covmat(3,1));
diagonal_2 = covmat(2,2) - ((covmat(2,3)./covmat(3,3))*covmat(3,2));
covmat_red = [diagonal_1, -0.9851*sqrt(diagonal_1.*diagonal_2); -0.9851*sqrt(diagonal_1.*diagonal_2), diagonal_2];
output_nom = mvnrnd(data(1,[1,2]), covmat_red, Ndata); % Generate the sythetic data from the input data  
output = [output_nom, ones(Ndata,1).*data(1,3)];

end

% Ensure the zero entries are set as zeros in the final synthetic data-set:
for j = 1:Ndata
output(j, idx) = 0;
end

%% End of function
end
