%% Data-processing for prediction of Fracture properties:

%% Load Raw Data:

[num,txt,data_raw] = xlsread('INCEFA_for_AI_v02.xls');

%% Define description function:
% This function returns a column vector of the data length, its mean, standard
% deviation, min value, 25-percentile, 50-percentile, 75-percentile, and
% max value.

des_model = @(x) [length(x), mean(x), std(x), min(x), prctile(x, [25, 50, 75]), max(x)]';
% Note: The input x is a row/column vector
%% Create Table of Raw Data:

% Extract key data of target feature from raw data:
N25 = num(:,89);      % Identify N25 as a target feature
for i = 1:length(N25) % Truncate N25 at 10000 - set all N25 > 10000 as = 10000
if N25(i) > 10000
N25(i) = 10000;
else
N25(i) = N25(i);
end
end

% Extract key data of input feature from raw data:
material_no = num(:,5); material_x = grp2idx(material_no); 
norm_strain_range2 = num(:,79); norm_mean_strain = num(:,80); 
norm_surface_roughness_Rt = num(:,82); 
norm_surface_roughness_Rt(isnan(norm_surface_roughness_Rt)) = 0; % Set all NaN entries as 0
norm_diameter = num(:,83); norm_hold_times = num(:,84); norm_environmental = num(:,85); 
norm_strain_rate = num(:,86); norm_temperature = num(:,87); 

% Create table of data:
des = {'Material_no', 'Norm_Strain_Range2', 'Norm_Mean_Strain', 'Norm_Surface_Roughness_Rt', ...
       'Norm_Diameter', 'Norm_Hold_Times', 'Norm_Environmental', 'Norm_Strain_Rate',...
       'Norm_Temperature', 'N25'};
table_data = table(material_no, norm_strain_range2, norm_mean_strain, norm_surface_roughness_Rt, ...
                 norm_diameter, norm_hold_times, norm_environmental, norm_strain_rate, norm_temperature, ...
                 N25, 'VariableNames', des);
             
% Create table of input feature description/statistics:
description = {'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'}';
des_material_x = des_model(material_x);                            % Material no. description (Input feature)
des_norm_strain_range = des_model(norm_strain_range2);             % Norm. strain range_2 description (Input feature)
des_norm_mean_strain = des_model(norm_mean_strain);                % Norm. mean strain description (Input feature)
des_norm_surface_roughness = des_model(norm_surface_roughness_Rt); % Norm. surface roughness Rt description (Input feature)
des_norm_diameter = des_model(norm_diameter);                      % Norm. diameter description (Input feature)
des_hold_times = des_model(norm_hold_times);                       % Norm. hold times description (Input feature)
des_norm_env = des_model(norm_environmental);                      % Norm. environmental description (Input feature)
des_norm_strain_rate = des_model(norm_strain_rate);                % Norm. strain rate description (Input feature)
des_norm_temperature = des_model(norm_temperature);                % Norm. temperature description (Input feature)
des_N25 = des_model(N25);                                          % N25 description (Output/Target feature)

table_des = table(description, des_material_x, des_norm_strain_range, ...
                  des_norm_mean_strain, des_norm_surface_roughness, ...
                  des_norm_diameter, des_hold_times, des_norm_env,...
                  des_norm_strain_rate, des_norm_temperature, des_N25,...
                  'VariableNames', {'Description', des{:,:}});
              
%% Save the Re-processed data:

% Save reprocessed data for INCEFA:
writetable(table_data, 'INCEFA_data_processed.csv') 

%% Input features for prediction of N25 properties:

% Create a correlation table of input features vs output features:
corr_array = corr(table2array(table_data(:,2:end)), 'Type', 'Spearman');
corr_table = array2table(corr_array, 'VariableNames', {des{2:end}}, 'RowNames', {des{2:end}});
corr_table(9,:) = []; corr_table(:,1:8) = []; % Rows: Input features, Columns: Output features

% Plot the parallel plots:
figure; 
mat = table2array(corr_table);
hold on; box on; grid on;
des1 = {'Material No.', 'Norm. Strain Range2', 'Norm. Mean Strain', 'Norm. Surface Roughness Rt', ...
       'Norm. Diameter', 'Norm. Hold Times', 'Norm. Environmental', 'Norm. Strain Rate',...
       'Norm. Temperature', 'N25'};
plot([1,length(mat)], [0, 0], 'k--', 'handlevisibility', 'off')
parallelcoords(mat', 'Labels', {des1{[2:9]}}, 'color', 'r', 'linewidth', 2); 
xlim([1,length(mat)]); xticks([1:length(mat)]); set(gca,'XTickLabel',{des1{[2:9]}}); 
xlabel('Input Features'); ylabel('Spearman Corr.')
legend('N25', 'linewidth',2, 'Location', 'Southeast')
set(gca,'Fontsize',10);

% Rank the Sensitivity according to the magnitude of Spearman Corr:
mat = abs(table2array(corr_table)); tab = array2table(mat, 'VariableNames', {des{10}}, 'RowNames', {des{[2:9]}});
sens_table = sortrows(tab,{des{10}},'descend'); 

% Plot scatterplot disgrams of input features vs output features:
figure;
table_array = table2array(table_data(:,2:end));
f = 10;  % Fontsize
idx = [1:8]; % Index of input features
for i = 1:length(idx)
subplot(3,3,i)
hold on; box on; grid on;
scatter(table_array(:,idx(i)), table_array(:,9),13, 'filled')
xlabel(des{idx(i)+1}, 'Interpreter', 'latex'); ylabel('N25', 'Interpreter', 'latex');
set(gca, 'Fontsize', f)
end
