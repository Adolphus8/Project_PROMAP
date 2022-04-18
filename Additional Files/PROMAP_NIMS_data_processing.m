%% Data-processing for prediction of Material properties:

%% Load Raw Data:

% Load Raw data for Creep rupture properties:
[num1a,txt1a,~] = xlsread('Power_plant_steels_NIMS_040903_database.xls','creeprupture');

% Load Raw data for Tensile properties:
[num1b,txt1b,~] = xlsread('Power_plant_steels_NIMS_040903_database.xls','tensile');

% Load Raw data for Hardness properties:
[num1c,txt1c,~] = xlsread('Power_plant_steels_NIMS_040903_database.xls','hardness');
                
% Load Raw data for Material Chemical composition:
[num2,txt2,~] = xlsread('Power_plant_steels_NIMS_040903_database.xls','Composition');

%% Create data table of Raw Data:

% Extract key features from raw data for Creep rupture properties:
len1 = 9966; % Row number of last entry from raw data
material_1 = txt1a(2:len1,1); cast_code_1 = txt1a(2:len1,2); 
stress = num1a(2:end,1);stress(9964:9965,1) = NaN; 
temperature = num1a(2:end,2); temperature(9964:9965,1) = NaN;
fracture = num1a(2:end,21); fracture(9964:9965,1) = NaN;
elongation = num1a(2:end,22); elongation(9964:9965,1) = NaN;
RA = num1a(2:end,23); RA(9964:9965,1) = NaN;

table_raw1a = table(material_1, cast_code_1, stress, temperature, fracture, elongation, RA);

% Extract key raw data from raw data for Tensile properties:
len1 = 3346; % Row number of last entry from raw data
material_1 = txt1b(3:len1,1); cast_code_1 = txt1b(3:len1,2); 
temperature = num1b(3:end,1); temperature(3343:3344,1) = NaN; 
PS_02 = num1b(3:end,7); PS_02(3343:3344,1) = NaN;
UTS = num1b(3:end,9); UTS(3343:3344,1) = NaN;
elongation = num1b(3:end,10); elongation(3343:3344,1) = NaN;
RofA = num1b(3:end,11); RofA(3343:3344,1) = NaN;

table_raw1b = table(material_1, cast_code_1, temperature, PS_02, UTS, elongation, RofA);

% Extract key raw data from raw data for Hardness properties:
len1 = 336; % Row number of last entry from raw data
material_1 = txt1c(2:len1,1); cast_code_1 = txt1c(2:len1,2); 
HRB = num1c(:,1); 

table_raw1c = table(material_1, cast_code_1, HRB);

% Extract key raw data from raw data for Material Chemical Composition:
len2 = 336; % Row number of last entry from raw data 2
material_1 = txt2(2:len2,1); material_x2 = grp2idx(material_1);
cast_code_1 = txt2(2:len2,3); cast_code_x2 = grp2idx(cast_code_1);
num2(isnan(num2)) = 0; % Set all NaN entries as 0
Comp_C = num2(:,1); Comp_Cr = num2(:,2); Comp_Co = num2(:,3); Comp_Al = num2(:,4); 
Comp_Ti = num2(:,5); Comp_Mo = num2(:,6); Comp_W = num2(:,7); Comp_Nb = num2(:,8); 
Comp_Fe = num2(:,9); Comp_B = num2(:,10); Comp_Zr = num2(:,11); Comp_V = num2(:,12); 
Comp_Ni = num2(:,13); Comp_Si = num2(:,14); Comp_Mn = num2(:,15); Comp_Cu = num2(:,16); 
Comp_S = num2(:,17); Comp_Pb = num2(:,18); Comp_Ta = num2(:,19); Comp_P = num2(:,20); 
Comp_Hf = num2(:,21); Comp_Pb1 = num2(:,22); Comp_Bi = num2(:,23); Comp_Sn = num2(:,24); 
Comp_Sb = num2(:,25); Comp_Zn = num2(:,26); Comp_Ag = num2(:,27); Comp_As = num2(:,28); 
Comp_Te = num2(:,29); Comp_Cd = num2(:,30); Comp_O = num2(:,31); Comp_N = num2(:,32); 
Comp_O2 = num2(:,33); Comp_N2 = num2(:,34); Comp_Hf1 = num2(:,35); Comp_Y = num2(:,36); 
Comp_Sc = num2(:,37); Comp_Be = num2(:,38); Comp_Mg = num2(:,39); Comp_NbTa = num2(:,40); 
Ceq = num2(:,41);

table_raw2 = table(material_1, cast_code_1, Comp_C, Comp_Cr, Comp_Co, Comp_Al, Comp_Ti, Comp_Mo, ...
                   Comp_W, Comp_Nb, Comp_Fe, Comp_B, Comp_Zr, Comp_V, Comp_Ni, Comp_Si, Comp_Mn, ...
                   Comp_Cu, Comp_S, Comp_Pb, Comp_Ta, Comp_P, Comp_Hf, Comp_Pb1, Comp_Bi, Comp_Sn, ...
                   Comp_Sb, Comp_Zn, Comp_Ag, Comp_As, Comp_Te, Comp_Cd, Comp_O, Comp_N, Comp_O2, ...
                   Comp_N2, Comp_Hf1, Comp_Y, Comp_Sc, Comp_Be, Comp_Mg, Comp_NbTa, Ceq);

%% Data Cleaning and Re-processing:

% Re-process data for Creep rupture properties:
table_raw1a = rmmissing(table_raw1a);     % Removes rows with missing/NaN entries
table1a_nom = outerjoin(table_raw1a,table_raw2(:,2:43),'MergeKeys', true); 
table1a_nom = rmmissing(table1a_nom);     % Removes rows with missing/NaN entries
mat = table2array(table1a_nom(:,1:2));    % Identify first 2 columns - Material name & Cast code
material_x = grp2idx(mat(:,1)); cast_code = grp2idx(mat(:,2));       % Label encode the Material name & Cast code
out1a = refine([table(material_x,cast_code), table1a_nom(:,3:end)]); % Drop all columns with only zero entries
table1a = out1a.table;

% Re-process data for Tensile properties:
table_raw1b = rmmissing(table_raw1b);     % Removes rows with missing/NaN entries
table1b_nom = outerjoin(table_raw1b,table_raw2(:,2:43),'MergeKeys', true); 
table1b_nom = rmmissing(table1b_nom);     % Removes rows with missing/NaN entries
mat = table2array(table1b_nom(:,1:2));    % Identify first 2 columns - Material name & Cast code
material_x = grp2idx(mat(:,1)); cast_code = grp2idx(mat(:,2));         % Label encode the Material name & Cast code
out1b = refine([table(material_x,cast_code), table1b_nom(:,3:end)]); % Drop all columns with only zero entries
table1b = out1b.table;

% Re-process data for Hardness properties:
table_raw1c = rmmissing(table_raw1c);     % Removes rows with missing/NaN entries
table1c_nom = outerjoin(table_raw1c,table_raw2(:,2:43),'MergeKeys', true); 
table1c_nom = rmmissing(table1c_nom);     % Removes rows with missing/NaN entries
mat = table2array(table1c_nom(:,1:2));    % Identify first 2 columns - Material name & Cast code
material_x = grp2idx(mat(:,1)); cast_code = grp2idx(mat(:,2));         % Label encode the Material name & Cast code
out1c = refine([table(material_x,cast_code), table1c_nom(:,3:end)]); % Drop all columns with only zero entries
table1c = out1c.table; table1c(74:85,:) = [];

%% Save the Re-processed data:

% Save reprocessed data for Creep rupture properties:
writetable(table1a, 'Creep_rupture_data_processed.csv') 

% Save reprocessed data for Tensile properties:
writetable(table1b, 'Tensile_data_processed.csv') 

% Save reprocessed data for Hardness properties:
writetable(table1c, 'Hardness_data_processed.csv') 

%% Data Analysis - Correlation Analysis:
%% Input features for prediction of Creep-rupture properties:

% Create a correlation table of input features vs output features:
table_array = table2array(table1a); indicator1a = out1a.indicator;
idx = [3:26]; corr1a = corr(table_array(:,idx), 'Type', 'Spearman');
des = {'Material_x', 'Cast_code', 'Stress', 'Temperature', 'Fracture', 'Elongation', 'RA', 'C', 'Cr', 'Co', 'Al',...
       'Ti', 'Mo', 'W', 'Nb', 'Fe', 'B', 'V', 'Ni', 'Si', 'Mn', 'Cu', 'S', 'P', 'N', 'NbTa'};
corr_table1a = array2table(corr1a, 'VariableNames', {des{idx}}, 'RowNames', {des{idx}});
corr_table1a([3:5],:) = []; corr_table1a(:,[1:2,6:end]) = []; % Rows: Input features, Columns: Output features

% Plot the parallel plots:
figure; 
mat = table2array(corr_table1a);
hold on; box on; grid on; col = {'r', 'g', 'b'};
plot([1,length(mat)], [0, 0], 'k--', 'handlevisibility', 'off')
for i = 1:3
parallelcoords(mat(:,i)', 'Labels', {des{[3,4,8:end]}}, 'color', col{i}, 'linewidth', 2);
end
xlim([1,length(mat)]); xticks([1:length(mat)]); set(gca,'XTickLabel',{des{[3,4,8:end]}}); 
xlabel('Input Features'); ylabel('Spearman Corr.')
legend('Fracture', 'Elongation', 'RA', 'linewidth',2, 'Location', 'Southeast')
set(gca,'Fontsize',13);

% Rank the Sensitivity according to the magnitude of Spearman Corr:
kdx = 1; % Sensitivity ranking results of input features for: 1 - Fracture; 2 - Elongation; 3 - RA.
out = {'Fracture', 'Elongation', 'RA'};
mat = abs(table2array(corr_table1a)); tab = array2table(mat, 'VariableNames', {des{5:7}}, 'RowNames', {des{[3,4,8:end]}});
sens_table1 = sortrows(tab(:,kdx),out{kdx},'descend'); 

% Plot scatterplot disgrams of input features vs output features:
figure;
kdx = 2; % Chosen output features: 1 - Fracture; 2 - Elongation; 3 - RA.
f = 16;  % Fontsize
%idx = [3,4,8:26]; % Index of input features
idx = [3,4,8:17];
for i = 1:length(idx)
%subplot(3,7,i)
subplot(3,4,i)
hold on; box on; grid on;
scatter(table_array(:,idx(i)), table_array(:,kdx+4),13, 'filled')
xlabel(des{idx(i)}, 'Interpreter', 'latex'); ylabel(out{kdx}, 'Interpreter', 'latex');
set(gca, 'Fontsize', f)
end

%% Input features for prediction of Tensile properties:

% Create a correlation table of input features vs output features:
table_array = table2array(table1b); indicator1b = out1b.indicator;
idx = [3:26]; corr1b = corr(table_array(:,idx), 'Type', 'Spearman');
des = {'Material_x', 'Cast_code', 'Temperature', 'PS02', 'UTS', 'Elongation', 'RA', 'C', 'Cr', 'Co', 'Al',...
       'Ti', 'Mo', 'W', 'Nb', 'Fe', 'B', 'V', 'Ni', 'Si', 'Mn', 'Cu', 'S', 'P', 'N', 'NbTa'};
corr_table1b = array2table(corr1b, 'VariableNames', {des{idx}}, 'RowNames', {des{idx}});
corr_table1b([2:5],:) = []; corr_table1b(:,[1,6:end]) = []; % Rows: Input features, Columns: Output features

% Plot the parallel plots:
figure; 
mat = table2array(corr_table1b);
hold on; box on; grid on; col = {'r', 'g', 'b', 'k'};
plot([1,length(mat)], [0, 0], 'k--', 'handlevisibility', 'off')
for i =1:4
parallelcoords(mat(:,i)', 'Labels', {des{[3,8:end]}}, 'color', col{i}, 'linewidth', 2); 
end
xlim([1,length(mat)]); xticks([1:length(mat)]); set(gca,'XTickLabel',{des{[3,8:end]}}); 
xlabel('Input Features'); ylabel('Spearman Corr.')
legend('PS02', 'UTS', 'Elongation', 'RA', 'linewidth',2, 'Location', 'Southeast')
set(gca,'Fontsize',13);

% Rank the Sensitivity according to the magnitude of Spearman Corr:
kdx = 4; % Sensitivity ranking results of input features for: 1 - PS_02; 2 - UTS; 3 - Elong; 4 - RofA.
out = {'PS02', 'UTS', 'Elongation', 'RA'};
mat = abs(table2array(corr_table1b)); tab = array2table(mat, 'VariableNames', {des{4:7}}, 'RowNames', {des{[3,8:end]}});
sens_table2 = sortrows(tab(:,kdx),out{kdx},'descend'); 

% Plot scatterplot diagrams of input features vs output features:
figure;
kdx = 4; % Chosen output features: 1 - PS02; 2 - UTS; 3 - Elongation; 4 - RA.
f = 16;  % Fontsize
%idx = [3,8:26]; % Index of input features
idx = [3,8:18];
for i = 1:length(idx)
%subplot(4,5,i)
subplot(3,4,i)
hold on; box on; grid on;
scatter(table_array(:,idx(i)), table_array(:,kdx+3),13, 'filled')
xlabel(des{idx(i)}, 'Interpreter', 'latex'); ylabel(out{kdx}, 'Interpreter', 'latex');
set(gca, 'Fontsize', f)
end

%% Input features for prediction of Hardness properties:

% Create a correlation table of input features vs output features:
table_array = table2array(table1c); indicator1c = out1c.indicator;
idx = [3:21]; corr1c = corr(table_array(:,idx), 'Type', 'Spearman');
des = {'Material_x', 'Cast_code', 'HRB', 'C', 'Cr', 'Co', 'Al',...
       'Ti', 'Mo', 'W', 'Nb', 'Fe', 'B', 'V', 'Ni', 'Si', 'Mn', 'Cu', 'S', 'P', 'N', 'NbTa'};
corr_table1c = array2table(corr1c, 'VariableNames', {des{idx}}, 'RowNames', {des{idx}});
corr_table1c(1,:) = []; corr_table1c(:,[2:end]) = []; % Rows: Input features, Columns: Output features

% Plot the parallel plots:
figure; 
mat = table2array(corr_table1c);
hold on; box on; grid on;
plot([1,length(mat)], [0, 0], 'k--', 'handlevisibility', 'off')
parallelcoords(mat', 'Labels', {des{[4:end-1]}}, 'color', 'r', 'linewidth', 2); 
xlim([1,length(mat)]); xticks([1:length(mat)]); set(gca,'XTickLabel',{des{[4:end-1]}}); 
xlabel('Input Features'); ylabel('Spearman Corr.')
legend('HRB', 'linewidth',2, 'Location', 'Southeast')
set(gca,'Fontsize',13);

% Rank the Sensitivity according to the magnitude of Spearman Corr:
mat = abs(table2array(corr_table1c)); tab = array2table(mat, 'VariableNames', {des{3}}, 'RowNames', {des{[4:21]}});
sens_table3 = sortrows(tab,{des{3}},'descend'); 

% Plot scatterplot disgrams of input features vs output features:
figure;
kdx = 1; % Chosen output features: 1 - PS_02; 2 - UTS; 3 - Elong; 4 - RofA.
f = 10;  % Fontsize
idx = [4:21]; % Index of input features
for i = 1:length(idx)
subplot(4,5,i)
hold on; box on; grid on;
scatter(table_array(:,idx(i)), table_array(:,3),13, 'filled')
xlabel(des{idx(i)}, 'Interpreter', 'latex'); ylabel('HRB', 'Interpreter', 'latex');
set(gca, 'Fontsize', f)
end
