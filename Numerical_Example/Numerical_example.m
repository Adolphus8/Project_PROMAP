%% Cantilever Model:
% Source: https://www.sfu.ca/~ssurjano/canti.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define the fixed parameters:

L = 2.5;           % Length of cantilever beam [m]
E = 2.0e+9;        % Young's Modulus of cantilever beam [N/m^2]
T = 0.20;          % Thickness of the cantilever beam [m]
W = 0.15;          % Width of the cantilever beam [m]

%% Cantilever beam model:
% x(:,1) - Horizontal load [N]
% x(:,2) - Vertical load [N]

D = @(x) ((4.*L^3)./(E.*W.*T)).*sqrt((x(:,1)./(W.^2)).^2 + ((x(:,2)./(T.^2)).^2));

%% Applied Load details:
% x(:,1) ~ [300, 500] U [600, 700] N; 
% x(:,2) ~ [800, 900] U [1000, 1200] N;
% Measurement uncertainty ~ 0.0003m

%% Case 1: When x(:,1) and x(:,2) are independent

% Generate some data:
Nsamps = 30;

sigma_mea = 0.0003; % Measurement error [m]

theta1_data = theta1_rnd(Nsamps); theta2_data = theta2_rnd(Nsamps);
input_data = [theta1_data, theta2_data];
measurement = D(input_data) + sigma_mea*randn(Nsamps,1);
data = [input_data, measurement];

%%
% Plot Scatterplots of the data:
des = {'$\theta_{1}$ $[N]$', '$\theta_{2}$ $[N]$', '$D$ $[m]$'}; id = 10; 

data_exp = data([1:id],:); data_val = data([id+1:end],:);

figure; s = 18; f = 18;
subplot(2,2,1)
hold on; box on; grid on;
scatter(data_exp(:,1), data_exp(:,3), s, 'g', 'filled')
scatter(data_val(:,1), data_val(:,3), s, 'b', 'filled')
xlabel(des{1}, 'Interpreter', 'latex'); ylabel(des{3}, 'Interpreter', 'latex')
set(gca, 'Fontsize', f)
subplot(2,2,2)
hold on; box on; grid on;
scatter(data_exp(:,2), data_exp(:,3), s, 'g', 'filled')
scatter(data_val(:,2), data_val(:,3), s, 'b', 'filled')
xlabel(des{2}, 'Interpreter', 'latex'); ylabel(des{3}, 'Interpreter', 'latex')
set(gca, 'Fontsize', f)
subplot(2,2,3)
hold on; box on; grid on;
scatter(data_exp(:,1), data_exp(:,2), s, 'g', 'filled')
scatter(data_val(:,1), data_val(:,2), s, 'b', 'filled')
xlabel(des{1}, 'Interpreter', 'latex'); ylabel(des{2}, 'Interpreter', 'latex')
legend('Observed data', 'Validation data', 'linewidth', 2)
set(gca, 'Fontsize', f)

%% Save the data:
save('Validation_exercise')

%% ANN Training with Exp data:

input_feature = data([1:10],1:2); target_feature = data([1:10],3);
input_feature_real = data([11:30],1:2); target_feature_real = data([11:30],3);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100;

% Train the ANNs:
fprintf('Train ANN1 \n')
tic;
[net1,tr1] = fit_NN(input_feature, target_feature, 9); 
time1 = toc;
y_pred = transpose(net1(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net1(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN1_Val_Ex3', 'net1', 'tr1', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN2 \n')
tic;
[net2,tr2] = fit_NN(input_feature, target_feature, 10); 
time1 = toc;
y_pred = transpose(net2(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net2(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN2_Val_Ex3', 'net2', 'tr2', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN3 \n')
tic;
[net3,tr3] = fit_NN(input_feature, target_feature, 11); 
time1 = toc;
y_pred = transpose(net3(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net3(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN3_Val_Ex3', 'net3', 'tr3', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN4 \n')
tic;
[net4,tr4] = fit_NN(input_feature, target_feature, 12); 
time1 = toc;
y_pred = transpose(net4(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net4(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN4_Val_Ex3', 'net4', 'tr4', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN5 \n')
tic;
[net5,tr5] = fit_NN(input_feature, target_feature, 13); 
time1 = toc;
y_pred = transpose(net5(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net5(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN5_Val_Ex3', 'net5', 'tr5', 'time1', 'r2_score_ver', 'r2_score_val')

%% ABMS Analysis I:

cse = 2; % Take values either 1 (verification) and 2 (validation).

des = {'$\theta_{1}$ $[N]$', '$\theta_{2}$ $[N]$', '$D$ $[m]$'};

figure;

load('ANN1_Val_Ex3.mat', 'net1'); net1 = net1;
load('ANN2_Val_Ex3.mat', 'net2'); net2 = net2;
load('ANN3_Val_Ex3.mat', 'net3'); net3 = net3;
load('ANN4_Val_Ex3.mat', 'net4'); net4 = net4;
load('ANN5_Val_Ex3.mat', 'net5'); net5 = net5;

CXann = {net1, net2, net3, net4, net5};
input_feature = data_exp(:,[1:2]); target_feature = data_exp(:,3);
CalibrationDataset = [input_feature, target_feature];

if cse == 1
MTest = data(1:id, [1:2]);
elseif cse == 2
MTest = data(11:30, [1:2]);
end

% Create category vector to allocate category for each training data:
cat_vec = [1*ones(3,1); 2*ones(4,1); 3*ones(3,1)];

tic;
[Vrobust, Vlower, Vupper, MnetP, ~] = applyAdaptiveBMS('CXann',CXann,... % Cell array of imported networks (to look inside remember to use curly brakets)
    'Mcalibrationdata',CalibrationDataset,...   % Calibration Data - Synthetic data
    'alpha',1.96,...                            % Accuracy of the confidence interval (1.96 >> 95% confidence interval)
    'Minputdata',MTest,...                      % Input feature data from Experimental data for Tensile properties
    'threshold',0.90,...                        % Value of posterior under which the probability punctual value is substituted by averaged value    
    'Lgraph',false,...                          % do you want a graphical representation of the solution? 
    'Lsort',false,...                           % do you want your graph to show the results sorted?
    'Category',cat_vec,...                      % categorising the output calibration data
    'Sprior','uniform',...                      % choose between 'uniform' or 'gm'
    'Sposterior','gm');                         % choose betweeen 'empirical' or 'gm'
timeABMS = toc;
Vlower(Vlower < 0) = 0;

col = [0.91, 0.41, 0.17];
hold on; box on; grid on; idx = length(MTest);
h = fill([(1:idx)';flipud((1:idx)')],[Vupper(1:idx);flipud(Vlower(1:idx))],'y','FaceAlpha',0.3,'EdgeColor','y', 'handlevisibility', 'off');
plot([(1:idx)]', Vlower(1:idx), 'color', col, 'linewidth', 1);
plot([(1:idx)]', Vupper(1:idx), 'color', col, 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vrobust(1:idx), 'k-- +', 'linewidth', 1)
if cse == 1
scatter([(1:idx)]', data(1:id,3), 18, 'b', 'filled'); 
xticks([1:size(data(1:id,3),1)]); xlim([1, size(data(1:id,3),1)]); xlabel('Observed Data no.');
legend('95% Confidence bounds', 'Robust predictions', 'Observed data', 'linewidth', 2)
elseif cse == 2
scatter([(1:idx)]', data(id+1:end,3), 18, 'b', 'filled');
xticks([1:size(data(id+1:end,3),1)]); xlim([1, size(data(id+1:end,3),1)]); xlabel('Validation Data no.');
legend('95% Confidence bounds', 'Robust predictions', 'Validation data', 'linewidth', 2)
end
ylabel(des{3}, 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)

%% Enhance the data-set

Ndata = 100;                     % The no. of synthetic data to generate
pc = 1;                          % Percentage perturbation about the original chosen data

% Construct the correlation matrix:
cormat = corrcoef(data_exp(:,[2,1,3])); 
des = {'F_x', 'F_y', 'D'};
cormat_table = array2table(cormat, 'VariableNames', des, 'RowNames', des);

figure; 
f = 25; des = {'F_x', 'F_y', 'D'};
imagesc(cormat); colorbar;
set(gca,'YTick', [1:size(data_exp,1)], 'YTickLabel', des, 'Fontsize', f);
set(gca,'XTick', [1:size(data_exp,1)], 'XTickLabel', des, 'Fontsize', f);

% Compute the partial correlations:
rho1 = partialcorr(data_exp(:,[2,3]),data_exp(:,1));
rho2 = partialcorr(data_exp(:,[1,3]),data_exp(:,2));
rho3 = partialcorr(data_exp(:,[1,2]),data_exp(:,3));

%% Enhance the data-set:

syn_mat_1 = [enhance_v2(data_exp(1,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(1,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(1,:), Ndata, cormat, pc, 3)];
syn_mat_2 = [enhance_v2(data_exp(2,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(2,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(2,:), Ndata, cormat, pc, 3)];
syn_mat_3 = [enhance_v2(data_exp(3,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(3,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(3,:), Ndata, cormat, pc, 3)];
syn_mat_4 = [enhance_v2(data_exp(4,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(4,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(4,:), Ndata, cormat, pc, 3)];
syn_mat_5 = [enhance_v2(data_exp(5,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(5,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(5,:), Ndata, cormat, pc, 3)];
syn_mat_6 = [enhance_v2(data_exp(6,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(6,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(6,:), Ndata, cormat, pc, 3)];
syn_mat_7 = [enhance_v2(data_exp(7,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(7,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(7,:), Ndata, cormat, pc, 3)];
syn_mat_8 = [enhance_v2(data_exp(8,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(8,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(8,:), Ndata, cormat, pc, 3)];
syn_mat_9 = [enhance_v2(data_exp(9,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(9,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(9,:), Ndata, cormat, pc, 3)];
syn_mat_10 = [enhance_v2(data_exp(10,:), Ndata, cormat, pc, 1); enhance_v2(data_exp(10,:), Ndata, cormat, pc, 2); enhance_v2(data_exp(10,:), Ndata, cormat, pc, 3)];

syn_mat = [syn_mat_1; syn_mat_2; syn_mat_3; syn_mat_4; syn_mat_5; syn_mat_6; syn_mat_7; syn_mat_8; syn_mat_9; syn_mat_10];

% Construct the correlation matrix:
cormat_syn = corrcoef(syn_mat(:,[2,1,3])); 
des = {'F_x', 'F_y', 'D'};
cormat_table_syn = array2table(cormat_syn, 'VariableNames', des, 'RowNames', des);

figure; 
f = 18; des = {'F_x', 'F_y', 'D'};
imagesc(cormat_syn); colorbar;
set(gca,'YTick', [1:size(syn_mat,2)], 'YTickLabel', des, 'Fontsize', f);
set(gca,'XTick', [1:size(syn_mat,2)], 'XTickLabel', des, 'Fontsize', f);

% Plot the Violinplot: Verification
figure;
% Compute parameters for Violinplot:
data_mat_norm = normalize([data_exp; syn_mat]); % Normalize the data
for i=1:size(data_mat_norm,2)
[f, u, bb] = ksdensity(data_mat_norm([size(data_exp,1)+1:end],i));
f=f/max(f)*0.1; % Normalize the PDF
F(:,i)=f; U(:,i)=u; bw(:,i)=bb;
end
f = 18; s = 18;
hold on; box on; grid on;
des = {'$F_y$ $[N]$', '$F_x$ $[N]$', '$D$ $[m]$'};
for i = 1:size(data_mat_norm,2)
h = fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],'m','FaceAlpha',0.1,'EdgeColor','m', 'handlevisibility', 'off');
scatter(i.*ones(size(syn_mat,1),1), data_mat_norm([size(data_exp,1)+1:end],i), s, 'rx')
scatter(i.*ones(size(data_exp,1),1), data_mat_norm([1:size(data_exp,1)],i), s-5, 'b', 'filled')
end
xticks([1:3]); xticklabels(des); ylabel('Normalised value'); xlim([0,4])
legend('Synthetic data', 'Original data', 'linewidth', 2, 'location', 'southeast')
set(gca, 'Fontsize', f)

% Plot the Violinplot: Validation
figure;
% Compute parameters for Violinplot:
for i=1:size(syn_mat,2)
[f, u, bb] = ksdensity(syn_mat(:,i));
f=f/max(f)*0.1; % Normalize the PDF
F(:,i)=f; U(:,i)=u; bw(:,i)=bb;
end
f = 18; s = 18; des = {'$F_y$ $[N]$', '$F_x$ $[N]$', '$D$ $[m]$'};
for i = 1:3
subplot(3,1,i)
hold on; box on; grid on;
h = fill([U(:,i);flipud(U(:,i))], [F(:,i)+1;flipud(1-F(:,i))],'m','FaceAlpha',0.1,'EdgeColor','m', 'handlevisibility', 'off');
scatter(syn_mat(:,i), ones(size(syn_mat(:,i),1),1), s, 'rx')
scatter(data_val(:,i), ones(size(data_val(:,i),1),1), s-5, 'b', 'filled')
%scatter(data([id+1:end],i), ones(size(data([id+1:end],i),1),1), s-5, 'b', 'filled')
scatter(data([1:10],i), ones(size(data([1:10],i),1),1), s-5, 'g', 'filled')
xlabel(des{i}, 'Interpreter', 'latex'); yticklabels([]); set(gca, 'Fontsize', f)
end
legend('Synthetic data', 'Validation data', 'Observed data', 'linewidth', 2, 'location', 'southeast')

table_exp_data = array2table(data([1:10],:), 'VariableNames', {'Theta_1', 'Theta_2', 'D'});
table_val_data = array2table(data([11:end],:), 'VariableNames', {'Theta_1', 'Theta_2', 'D'});
table_syn_data = array2table(syn_mat, 'VariableNames', {'Theta_1', 'Theta_2', 'D'});

figure;
subplot(2,2,1)
hold on; box on; grid on;
scatter(syn_mat(:,1), syn_mat(:,3), s, 'r', 'filled')
scatter(data_exp(:,1), data_exp(:,3), s, 'g', 'filled')
scatter(data_val(:,1), data_val(:,3), s, 'b', 'filled')
xlabel(des{1}, 'Interpreter', 'latex'); ylabel(des{3}, 'Interpreter', 'latex')
set(gca, 'Fontsize', f)
subplot(2,2,2)
hold on; box on; grid on;
scatter(syn_mat(:,2), syn_mat(:,3), s, 'r', 'filled')
scatter(data_exp(:,2), data_exp(:,3), s, 'g', 'filled')
scatter(data_val(:,2), data_val(:,3), s, 'b', 'filled')
xlabel(des{2}, 'Interpreter', 'latex'); ylabel(des{3}, 'Interpreter', 'latex')
set(gca, 'Fontsize', f)
subplot(2,2,3)
hold on; box on; grid on;
scatter(syn_mat(:,1), syn_mat(:,2), s, 'r', 'filled')
scatter(data_exp(:,1), data_exp(:,2), s, 'g', 'filled')
scatter(data_val(:,1), data_val(:,2), s, 'b', 'filled')
xlabel(des{1}, 'Interpreter', 'latex'); ylabel(des{2}, 'Interpreter', 'latex')
legend('Synthetic data', 'Observed data', 'Validation data', 'linewidth', 2)
set(gca, 'Fontsize', f)

%% ANN Training with Synthetic data:

input_feature1 = syn_mat(:,1:2); target_feature1 = syn_mat(:,3);
input_feature = data([1:10],1:2); target_feature = data([1:10],3);
input_feature_real = data([11:30],1:2); target_feature_real = data([11:30],3);

% Model to compute R2-score in [%]:
score_model = @(data, model_output) (1 - (sum((data - model_output).^2)./sum((data - mean(data)).^2)))*100;

% Train the ANNs:
fprintf('Train ANN1 \n')
tic;
[net1,tr1] = fit_NN(input_feature1, target_feature1, 9); 
time1 = toc;
y_pred = transpose(net1(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net1(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
%save('ANN1_Val_Ex3b', 'net1', 'tr1', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN2 \n')
tic;
[net2,tr2] = fit_NN(input_feature1, target_feature1, 10); 
time1 = toc;
y_pred = transpose(net2(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net2(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN2_Val_Ex3b', 'net2', 'tr2', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN3 \n')
tic;
[net3,tr3] = fit_NN(input_feature1, target_feature1, 11); 
time1 = toc;
y_pred = transpose(net3(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net3(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN3_Val_Ex3b', 'net3', 'tr3', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN4 \n')
tic;
[net4,tr4] = fit_NN(input_feature1, target_feature1, 12); 
time1 = toc;
y_pred = transpose(net4(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net4(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN4_Val_Ex3b', 'net4', 'tr4', 'time1', 'r2_score_ver', 'r2_score_val')

fprintf('Train ANN5 \n')
tic;
[net5,tr5] = fit_NN(input_feature1, target_feature1, 13); 
time1 = toc;
y_pred = transpose(net5(input_feature_real')); r2_score_val = score_model(target_feature_real, y_pred);
y_pred = transpose(net5(input_feature')); r2_score_ver = score_model(target_feature, y_pred);
save('ANN5_Val_Ex3b', 'net5', 'tr5', 'time1', 'r2_score_ver', 'r2_score_val')

%% ABMS Analysis:

cse = 2; % Take values either 1 (verification) and 2 (validation).

des = {'$\theta_{1}$ $[N]$', '$\theta_{2}$ $[N]$', '$D$ $[m]$'};

figure;
load('ANN1_Val_Ex3b.mat', 'net1'); net1 = net1;
load('ANN2_Val_Ex3b.mat', 'net2'); net2 = net2;
load('ANN3_Val_Ex3b.mat', 'net3'); net3 = net3;
load('ANN4_Val_Ex3b.mat', 'net4'); net4 = net4;
load('ANN5_Val_Ex3b.mat', 'net5'); net5 = net5;

CXann = {net1, net2, net3, net4, net5};
input_feature = syn_mat(:,1:2); target_feature = syn_mat(:,3);
CalibrationDataset = [input_feature, target_feature];

if cse == 1
MTest = data(1:id, [1:2]);
elseif cse == 2
MTest = data(id+1:end, [1:2]);
end

% Create category vector to allocate category for each training data:
cv = 300;
cat_vec = [1.*ones(cv,1); 2.*ones(cv,1); 3.*ones(cv,1); 4.*ones(cv,1); 5.*ones(cv,1);...
           6.*ones(cv,1); 7.*ones(cv,1); 8.*ones(cv,1); 9.*ones(cv,1); 10.*ones(cv,1)];


tic;
[Vrobust2, Vlower2, Vupper2,MnetP2, ~] = applyAdaptiveBMS('CXann',CXann,... % Cell array of imported networks (to look inside remember to use curly brakets)
    'Mcalibrationdata',CalibrationDataset,...   % Calibration Data - Synthetic data
    'alpha',1.96,...                            % Accuracy of the confidence interval (1.96 >> 95% confidence interval)
    'Minputdata',MTest,...                      % Input feature data from Experimental data for Tensile properties
    'threshold',0.90,...                        % Value of posterior under which the probability punctual value is substituted by averaged value    
    'Lgraph',false,...                          % do you want a graphical representation of the solution? 
    'Lsort',false,...                           % do you want your graph to show the results sorted?
    'Category',cat_vec,...                      % categorising the output calibration data
    'Sprior','uniform',...                      % choose between 'uniform' or 'gm'
    'Sposterior','gm');                         % choose betweeen 'empirical' or 'gm'
timeABMS = toc;
Vlower2(Vlower2 < 0) = 0;

col = [0.91, 0.41, 0.17];
hold on; box on; grid on; idx = length(MTest);
h = fill([(1:idx)';flipud((1:idx)')],[Vupper2(1:idx);flipud(Vlower2(1:idx))],'y','FaceAlpha',0.3,'EdgeColor','y', 'handlevisibility', 'off');
plot([(1:idx)]', Vlower2(1:idx), 'color', col, 'linewidth', 1);
plot([(1:idx)]', Vupper2(1:idx), 'color', col, 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vrobust2(1:idx), 'k-- +', 'linewidth', 1)
if cse == 1
scatter([(1:idx)]', data(1:id,3), 18, 'b', 'filled'); 
xticks([1:size(data(1:id,3),1)]); xlim([1, size(data(1:id,3),1)]); xlabel('Observed Data no.');
legend('95% Confidence bounds', 'Robust predictions', 'Observed data', 'linewidth', 2)
elseif cse == 2
scatter([(1:idx)]', data(id+1:end,3), 18, 'b', 'filled');
xticks([1:size(data(id+1:end,3),1)]); xlim([1, size(data(id+1:end,3),1)]); xlabel('Validation Data no.');
legend('95% Confidence bounds', 'Robust predictions', 'Validation data', 'linewidth', 2)
end
ylabel(des{3}, 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)

%%

[kdx,sdx] = sort(data_val(:,3),'ascend');

idx = 20;
figure;
hold on; box on; grid on;
plot([(1:idx)]', Vrobust([sdx]), 'b-- o', 'linewidth', 1)
plot([(1:idx)]', Vlower([sdx]), 'color', 'c', 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vupper([sdx]), 'color', 'c', 'linewidth', 1, 'handlevisibility', 'off');
h1 = fill([(1:idx)';flipud((1:idx)')],[Vupper([sdx]);flipud(Vlower([sdx]))],'c','FaceAlpha',0.3,'EdgeColor','c');
plot([(1:idx)]', Vrobust2([sdx]), 'r-- o', 'linewidth', 1)
plot([(1:idx)]', Vlower2([sdx]), 'color', 'm', 'linewidth', 1, 'handlevisibility', 'off');
plot([(1:idx)]', Vupper2([sdx]), 'color', 'm', 'linewidth', 1, 'handlevisibility', 'off');
h1 = fill([(1:idx)';flipud((1:idx)')],[Vupper2([sdx]);flipud(Vlower2([sdx]))],'m','FaceAlpha',0.3,'EdgeColor','m');
scatter([(1:idx)]', data_val([sdx],3), 18, 'k', 'filled');
xticks([1:size(data(id+1:end,3),1)]); xlim([1, size(data(id+1:end,3),1)]); xlabel('Validation Data no.');
legend('Robust predictions (Obs. data)', '95% Confidence bounds (Obs. data)', 'Robust predictions (Syn. data)', '95% Confidence bounds (Syn. data)', 'Validation data', 'linewidth', 2)
ylabel(des{3}, 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)

figure
subplot(2,1,1)
hold on; box on; grid on;
plot([(1:idx)]', MnetP([sdx],5), '--ys', 'MarkerFaceColor','y', 'linewidth', 1)
plot([(1:idx)]', MnetP([sdx],1), '--rs', 'MarkerFaceColor','r', 'linewidth', 1)
plot([(1:idx)]', MnetP([sdx],2), '--bs', 'MarkerFaceColor','b', 'linewidth', 1)
plot([(1:idx)]', MnetP([sdx],3), '--gs', 'MarkerFaceColor','g', 'linewidth', 1)
plot([(1:idx)]', MnetP([sdx],4), '--ms', 'MarkerFaceColor','m', 'linewidth', 1)
ylabel('$P(M_\nu,y_\nu|\mathbf{D})$', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18); 
xlim([1,idx]); xticks([1:idx]); xlabel('Validation Data no.'); title('Trained with Observed data')

subplot(2,1,2)
hold on; box on; grid on;
plot([(1:idx)]', MnetP2([sdx],5), '--ys', 'MarkerFaceColor','y', 'linewidth', 1)
plot([(1:idx)]', MnetP2([sdx],1), '--rs', 'MarkerFaceColor','r', 'linewidth', 1)
plot([(1:idx)]', MnetP2([sdx],2), '--bs', 'MarkerFaceColor','b', 'linewidth', 1)
plot([(1:idx)]', MnetP2([sdx],3), '--gs', 'MarkerFaceColor','g', 'linewidth', 1)
plot([(1:idx)]', MnetP2([sdx],4), '--ms', 'MarkerFaceColor','m', 'linewidth', 1)
legend('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'linewidth', 2)
ylabel('$P(M_\nu,y_\nu|\mathbf{D})$', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18); 
xlim([1,idx]); xticks([1:idx]); xlabel('Validation Data no.'); title('Trained with Synthetic data')

%% Violin plot:

% Normalize the data:
jdx = [1:3];
data_mat_norm = normalize([data_exp(:,jdx);syn_mat(:,jdx);data_val(:,jdx)]);

% Compute parameters for Violinplot:
for i=1:length(jdx)
[f, u, bb] = ksdensity(data_mat_norm([11:3010],i));
f=f/max(f)*0.1; %normalize
F(:,i)=f; U(:,i)=u; bw(:,i)=bb;
end

% Plot the figure:
figure;
f = 18; s = 18;
hold on; box on; grid on;
for i = [2,1,3]
h = fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],'m','FaceAlpha',0.1,'EdgeColor','m', 'handlevisibility', 'off');
scatter(i.*ones(size(syn_mat,1),1), data_mat_norm([11:3010],i), s, 'rx')
scatter(i.*ones(size(data_val,1),1), data_mat_norm([3011:end],i), s-5, 'b', 'filled')
scatter(i.*ones(size(data_exp,1),1), data_mat_norm([1:10],i), s-5, 'g', 'filled')
end
xticks([1:3]); xticklabels({'F_x', 'F_y', 'D'}); ylabel('Normalised value'); xlim([0,4])
legend('Synthetic data', 'Validation data', 'Observed data', 'linewidth', 2, 'location', 'southeast')
set(gca, 'Fontsize', f); 
