function [output] = refine(table_input)
%% The "refine" function-handle serves to remove table columns consisting of all zeros. 
% The description is as follows:
%
% Function input(s):
% - table_input: The input table goes here whose rows (first dimension) are
% the data entries while the columns (second dimension) are the variables;
%
% Function output(s):
% - output.table:     The output table goes here where column(s) with only zero entries are removed;
% - output.indicator: The 1 x N vector of indicators denoting the no. of zero entries per column; 
%
%% The function details:

% First, convert the table into an array:
table_array = table2array(table_input);

% Next, create an indicator vector which would indicate the number of zeros entries per column:
zeros_indicator = zeros(1,size(table_array,2));

% Initiate the loop to commence the computation of zeros entries for each column:
for i = 1:size(table_array,2)       % Outer loop accounts for each column entry
vec = zeros(size(table_array,1),1); % Initiate empty vector to indicate the zero entries for the column j
for j = 1:size(table_array,1)       % Inner loop accounts for each row entry
if table_array(j,i) == 0            % If the row entry in that column is 0, set indicator as 1
vec(j) = 1;
else                                % Otherwise, set indicator as 0
vec(j) = 0;
end
end
zeros_indicator(i) = sum(vec);      % Sum the indicator vector to ontain the no. of zero entries for that column j
end

idx = find(zeros_indicator == size(table_array,1)); % Find all column(s) with all zero entries
table_input(:,idx') = [];                           % Drop columns in Data table which have all 0 entries
zeros_indicator(idx') = [];                         % Drop columns in indicator which have been dropped
output.table = table_input;                         % Obtain the output refined table
output.indicator = zeros_indicator;                 % Obtain the output zeros indicator for each column

%% End of function
end

