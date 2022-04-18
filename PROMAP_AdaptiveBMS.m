%% ABMS Analysis for Creep Rupture Properties Prediction:

% Load the synthetic data:
table1a = readtable('Creep_rupture_syndata_small.csv'); 
syndata_mat = table2array(table1a);
input_feature = syndata_mat(:, [1:4,8:26]); target_feature = syndata_mat(:, [5:7]);

% Load the experimental data:
table1b = readtable('Creep_rupture_data_processed.csv'); 
realdata_mat = table2array(table1b); kdx = find(realdata_mat(:,2) == 221);
input_feature_real = realdata_mat(kdx, [1:4,8:26]); target_feature_real = realdata_mat(kdx, [5:7]);


des = {'FT [hrs]', 'Elongation [$\%$]', 'RA [$\%$]'};

figure;
for tar =1:3
if tar == 1
load('ANN_Mat_Files/Creep_Rupture/ANN1_CreepRupture_Frac.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN2_CreepRupture_Frac.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN3_CreepRupture_Frac.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN4_CreepRupture_Frac.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN7_CreepRupture_Frac.mat', 'net'); net5 = net;
elseif tar == 2
load('ANN_Mat_Files/Creep_Rupture/ANN1_CreepRupture_Elong.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN2_CreepRupture_Elong.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN3_CreepRupture_Elong.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN4_CreepRupture_Elong.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN6_CreepRupture_Elong.mat', 'net'); net5 = net;
elseif tar == 3
load('ANN_Mat_Files/Creep_Rupture/ANN1_CreepRupture_RA.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN2_CreepRupture_RA.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN3_CreepRupture_RA.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN4_CreepRupture_RA.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Creep_Rupture/ANN6_CreepRupture_RA.mat', 'net'); net5 = net;
end

CXann = {net1, net2, net3, net4, net5};
CalibrationDataset = [input_feature, target_feature(:,tar)];
MTest = input_feature_real;

% Create category vector to allocate category for each training data:
cat_vec = [1.*ones(1000,1); 2.*ones(1000,1); 3.*ones(1000,1); 4.*ones(1000,1); 5.*ones(1000,1);...
           6.*ones(1000,1); 7.*ones(1000,1); 8.*ones(1000,1); 9.*ones(1000,1); 10.*ones(1000,1)];

% Run ABMS method:
tic;
[Vrobust, Vlower, Vupper] = applyAdaptiveBMS('CXann',CXann,... % Cell array of imported networks (to look inside remember to use curly brakets)
    'Mcalibrationdata',CalibrationDataset,...   % Calibration Data - Synthetic data
    'alpha',1.96,...                            % Accuracy of the confidence interval (1.96 >> 95% confidence interval)
    'Minputdata',MTest,...                      % Input feature data from Experimental data for Creep Rupture properties
    'threshold',0.90,...                        % Value of posterior under which the probability punctual value is substituted by averaged value    
    'Lgraph',false,...                          % do you want a graphical representation of the solution? 
    'Lsort',false,...                           % do you want your graph to show the results sorted?
    'Category',cat_vec,...                      % categorising the output calibration data
    'Sprior','uniform',...                      % choose between 'uniform' or 'gm'
    'Sposterior','gm');                         % choose betweeen 'empirical' or 'gm'
timeABMS(tar) = toc;
Vlower(Vlower < 0) = 0;

subplot(2,2,tar)
col = [0.91, 0.41, 0.17];
hold on; box on; grid on; idx = 13;
plot([(1:idx)]', Vlower(1:idx), 'color', col, 'linewidth', 1);
plot([(1:idx)]', Vupper(1:idx), 'color', col, 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vrobust(1:idx), 'k-- +', 'linewidth', 1)
scatter([(1:idx)]', target_feature_real([1:idx],tar), 18, 'b', 'filled');
xlabel('Data no.'); ylabel(des{tar}, 'Interpreter', 'latex'); xlim([1, size(target_feature_real,1)]); 
set(gca, 'Fontsize', 18)
xticks([1:size(target_feature_real,1)])
end
legend('95% Confidence bounds', 'Robust predictions', 'Experimental data', 'linewidth', 2)

%% ABMS Analysis for Tensile Properties Prediction:

% Load the synthetic data:
table1a = readtable('Tensile_syndata_small.csv'); 
syndata_mat = table2array(table1a);
input_feature = syndata_mat(:, [1:3,8:26]); target_feature = syndata_mat(:, [4:7]);

% Load the experimental data:
table1b = readtable('Tensile_data_processed.csv'); 
realdata_mat = table2array(table1b); kdx = find(realdata_mat(:,2) == 45);
input_feature_real = realdata_mat(kdx, [1:3,8:26]); target_feature_real = realdata_mat(kdx, [4:7]);

des = {'PS02 [Mpa]', 'UTS [MPa]', 'Elongation [$\%$]', 'RA [$\%$]'};

figure;
for tar = 1:4
if tar == 1
load('ANN_Mat_Files/Tensile/ANN1_Tensile_PS02.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Tensile/ANN2_Tensile_PS02.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Tensile/ANN3_Tensile_PS02.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Tensile/ANN4_Tensile_PS02.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Tensile/ANN5_Tensile_PS02.mat', 'net'); net5 = net;
elseif tar == 2
load('ANN_Mat_Files/Tensile/ANN1_Tensile_UTS.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Tensile/ANN2_Tensile_UTS.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Tensile/ANN3_Tensile_UTS.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Tensile/ANN4_Tensile_UTS.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Tensile/ANN5_Tensile_UTS.mat', 'net'); net5 = net;
elseif tar == 3
load('ANN_Mat_Files/Tensile/ANN1_Tensile_Elong.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Tensile/ANN2_Tensile_Elong.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Tensile/ANN3_Tensile_Elong.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Tensile/ANN4_Tensile_Elong.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Tensile/ANN6_Tensile_Elong.mat', 'net'); net5 = net;
elseif tar == 4
load('ANN_Mat_Files/Tensile/ANN1_Tensile_RofA.mat', 'net'); net1 = net;
load('ANN_Mat_Files/Tensile/ANN2_Tensile_RofA.mat', 'net'); net2 = net;
load('ANN_Mat_Files/Tensile/ANN3_Tensile_RofA.mat', 'net'); net3 = net;
load('ANN_Mat_Files/Tensile/ANN4_Tensile_RofA.mat', 'net'); net4 = net;
load('ANN_Mat_Files/Tensile/ANN5_Tensile_RofA.mat', 'net'); net5 = net;
end

CXann = {net1, net2, net3, net4, net5};
CalibrationDataset = [input_feature, target_feature(:,tar)];
MTest = input_feature_real;

% Create category vector to allocate category for each training data:
cat_vec = [1.*ones(1000,1); 2.*ones(1000,1); 3.*ones(1000,1); 4.*ones(1000,1); 5.*ones(1000,1);...
           6.*ones(1000,1); 7.*ones(1000,1); 8.*ones(1000,1); 9.*ones(1000,1); 10.*ones(1000,1)];

% Run ABMS method:
tic;
[Vrobust, Vlower, Vupper] = applyAdaptiveBMS('CXann',CXann,... % Cell array of imported networks (to look inside remember to use curly brakets)
    'Mcalibrationdata',CalibrationDataset,...   % Calibration Data - Synthetic data
    'alpha',1.96,...                            % Accuracy of the confidence interval (1.96 >> 95% confidence interval)
    'Minputdata',MTest,...                      % Input feature data from Experimental data for Tensile properties
    'threshold',0.90,...                        % Value of posterior under which the probability punctual value is substituted by averaged value    
    'Lgraph',false,...                          % do you want a graphical representation of the solution? 
    'Lsort',false,...                           % do you want your graph to show the results sorted?
    'Category',cat_vec,...                      % categorising the output calibration data
    'Sprior','uniform',...                      % choose between 'uniform' or 'gm'
    'Sposterior','gm');                         % choose betweeen 'empirical' or 'gm'
timeABMS(tar) = toc;
Vlower(Vlower < 0) = 0;

subplot(2,2,tar)
col = [0.91, 0.41, 0.17];
hold on; box on; grid on; idx = 8;
plot([(1:idx)]', Vlower(1:idx), 'color', col, 'linewidth', 1);
plot([(1:idx)]', Vupper(1:idx), 'color', col, 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vrobust(1:idx), 'k-- +', 'linewidth', 1)
scatter([(1:idx)]', target_feature_real([1:idx],tar), 18, 'b', 'filled');
xlabel('Data no.'); ylabel(des{tar}, 'Interpreter', 'latex'); xlim([1, size(target_feature_real,1)]); 
set(gca, 'Fontsize', 18)
xticks([1:size(target_feature_real,1)])
end
legend('95% Confidence bounds', 'Robust predictions', 'Experimental data', 'linewidth', 2)

