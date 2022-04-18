%% Train the ANN for Creep Rupture prediction:

% Load Synthetic data:
table1a = readtable('Creep_rupture_syndata_small.csv'); 
syndata_mat = table2array(table1a);
input_feature = syndata_mat(:, [1:4,8:26]); target_feature = syndata_mat(:, [5:7]);

% Load Experimental data:
table1b = readtable('Creep_rupture_data_processed.csv'); 
realdata_mat = table2array(table1b); kdx = find(realdata_mat(:,2) == 221);
input_feature_real = realdata_mat(kdx, [1:4,8:26]); target_feature_real = realdata_mat(kdx, [5:7]);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100; 

% Train the ANNs:
fprintf('Fit Creep Rupture Fracture \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN1_CreepRupture_Frac', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN2_CreepRupture_Frac', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN3_CreepRupture_Frac', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN4_CreepRupture_Frac', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 7); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN7_CreepRupture_Frac', 'net', 'tr', 'time1', 'r2_score')

fprintf('Fit Creep Rupture Elongation \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN1_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN2_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN3_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN4_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN5_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

fprintf('Fit Creep Rupture RA \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN1_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN2_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN3_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN4_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN5_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

%% Train Existing ANN by N. Prinja for Creep Rupture Properties prediction:
% To be trained with experimental data:
clear; clc;

table1a = readtable('Creep_rupture_data_processed.csv'); 
creep_mat = table2array(table1a);                        % Convert tabular data into array
tra = find(creep_mat(:,1) == 13); creep_mat(tra,:) = []; % Remove data whose composition is 0 for all elements
input_feature_real = creep_mat(:,[1:4,8:26]); target_feature_real = creep_mat(:,[5:7]);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100;

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,1), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN1_Prinja_CreepRupture_Fracture', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,1), 7); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN2_Prinja_CreepRupture_Fracture', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,2), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN1_Prinja_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,2), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN2_Prinja_CreepRupture_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,3), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN1_Prinja_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,3), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN2_Prinja_CreepRupture_RA', 'net', 'tr', 'time1', 'r2_score')

%% Train the ANN for Tensile Properties prediction
clear; clc;

table1a = readtable('Tensile_syndata_small.csv'); 
syndata_mat = table2array(table1a);
input_feature = syndata_mat(:, [1:3,8:26]); target_feature = syndata_mat(:, [4:7]);

% Load Experimental data:
table1b = readtable('Tensile_data_processed.csv'); 
realdata_mat = table2array(table1b); kdx = find(realdata_mat(:,2) == 45);
input_feature_real = realdata_mat(kdx, [1:3,8:26]); target_feature_real = realdata_mat(kdx, [4:7]);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100; 

% Train the ANNs:
fprintf('Fit Tensile PS02 \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN1_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN2_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN3_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN4_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,1), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN6_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

fprintf('Fit Tensile UTS \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN1_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN2_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN3_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN4_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,2), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN6_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

fprintf('Fit Tensile Elongation \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN1_Tensile_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN2_Tensile_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN3_Tensile_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN4_Tensile_Elong', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,3), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN5_Tensile_Elong', 'net', 'tr', 'time1', 'r2_score')

fprintf('Fit Tensile RofA \n')
tic;
[net,tr] = fit_NN(input_feature, target_feature(:,4), 1); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN1_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,4), 2); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN2_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,4), 3); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN3_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,4), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN4_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature, target_feature(:,4), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN6_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

%% Train Existing ANN by N. Prinja for Tensile Properties prediction:
% To be trained with experimental data:
clear; clc;

table1a = readtable('Tensile_data_processed.csv'); 
tensile_mat = table2array(table1a);                          % Convert tabular data into array
tra = find(tensile_mat(:,1) == 13); tensile_mat(tra,:) = []; % Remove data whose composition is 0 for all elements
input_feature_real = tensile_mat(:,[1:3,8:26]); target_feature_real = tensile_mat(:,[4:7]);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100;

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,1), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN1_Prinja_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,1), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,1), y_pred);
save('ANN2_Prinja_Tensile_PS02', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,2), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN1_Prinja_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,2), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,2), y_pred);
save('ANN2_Prinja_Tensile_UTS', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,3), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN1_Prinja_Tensile_Elongation', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,3), 5); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,3), y_pred);
save('ANN2_Prinja_Tensile_Elongation', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,4), 4); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN1_Prinja_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')

tic;
[net,tr] = fit_NN(input_feature_real, target_feature_real(:,4), 6); 
time1(1) = toc;
y_pred = transpose(net(input_feature_real')); r2_score = score_model(target_feature_real(:,4), y_pred);
save('ANN2_Prinja_Tensile_RofA', 'net', 'tr', 'time1', 'r2_score')
