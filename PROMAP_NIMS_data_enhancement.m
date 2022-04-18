%% Data-enhancement for prediction of Material properties:

%% Load Processed Data:

% Load processed data for Creep rupture properties:
table1a = readtable('Creep_rupture_data_processed.csv'); 
creep_mat = table2array(table1a);    % Convert tabular data into array

% Load processed data for Tensile properties:
table1b = readtable('Tensile_data_processed.csv'); 
tensile_mat = table2array(table1b);  % Convert tabular data into array

%% Perturbate the existing data (Enhancing the data-set):

Ndata = 10000;                      % The no. of synthetic data to generate
pc = 1;                             % Percentage perturbation about the original chosen data

%% Creep rupture properties: 

des = {'Material_x', 'Cast_code', 'Stress', 'Temperature', 'FT', 'Elongation',...
       'RA', 'C', 'Cr', 'Co', 'Al', 'Ti', 'Mo', 'W', 'Nb', 'Fe', 'B', 'V', 'Ni', 'Si',...
       'Mn', 'Cu', 'S', 'P', 'N', 'NbTa'};

% Identify the number of data that falls into the corresponding category of cast_code:
index = zeros(max(creep_mat(:,2)),1); % Create empty array to record number of data for a cast_code category
for i = 1:max(creep_mat(:,2))
idx = find(creep_mat(:,2) == i); 
index(i) = length(idx);
end
% Identify the category of cast_code with the lowest and highest data-set:
fdx_min = find(index == min(index)); fdx_max = find(index == max(index));

% Find all rows corresponding to cast_code with lowest data-set: 
idx = find(creep_mat(:,2) == fdx_min(1)); 
data = table1a(idx,:); data_mat = table2array(data);

% Define the correlation matrix of all features in the chosen category:
kdx = [3:7]; 
cormat = corrcoef(data_mat(:,kdx)); 
cormat_table = array2table(cormat, 'VariableNames', {des{kdx}}, 'RowNames', {des{kdx}});

figure; 
f = 18;
imagesc(cormat); colorbar;
set(gca,'YTick', [1:length(kdx)],'YTickLabel',{des{kdx}}, 'Fontsize', f);
set(gca,'XTick', [1:length(kdx)],'XTickLabel',{des{kdx}}, 'Fontsize', f);

% Enhance the data-set:
sel = sort(unique(data_mat(:,5)),'descend'); % Find unique values of the Fracture data
N = [2000,1500,1500,500*ones(1,10)];
for i = 1:length(sel)
jdx = find(data_mat(:,5) == sel(i));
syn_nom =  enhance(data_mat(jdx(1), kdx), N(i), cormat, pc); % Select a data entry to populate
if i==1
syn_mat = syn_nom;
else
syn_mat = [syn_mat; syn_nom];
end
end

% Create table of synthetic data:
syn_data = array2table([repmat(data_mat(1,[1:2]),Ndata,1), syn_mat, repmat(data_mat(1,[8:end]),Ndata,1)],...
           'VariableNames', des);
writetable(syn_data, 'Creep_rupture_syndata_small.csv') 

% Plot the category of key input feature values:
figure;
for i = 1:length(kdx)
subplot(2,3,i)
hold on; box on; grid on;
f = 15; s = 18;
scatter(kdx(i).*ones(Ndata,1), syn_mat(:,i), s, 'rx')
scatter(kdx(i).*ones(length(idx),1), data_mat(:,kdx(i)), s+3, 'b')
xlabel(des{kdx(i)}); ylabel('Feature value');
set(gca, 'Fontsize', f); set(gca,'XTickLabel',[]);
end
legend('Synthetic data', 'Experimental data', 'linewidth', 2)

% Compute parameters for Violinplot:
data_mat_norm = normalize([data_mat(:,[3:7]); syn_mat]); % Normalize the data
for i=1:length(kdx)
[f, u, bb] = ksdensity(data_mat_norm([size(data_mat,1)+1:end],i));
f=f/max(f)*0.1; % Normalize the PDF
F(:,i)=f; U(:,i)=u; bw(:,i)=bb;
end

% Plot the Violinplot:
figure;
f = 18; s = 18;
hold on; box on; grid on;
for i = 1:length(kdx)
h = fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],'m','FaceAlpha',0.1,'EdgeColor','m', 'handlevisibility', 'off');
scatter(i.*ones(size(syn_mat,1),1), data_mat_norm([size(data_mat,1)+1:end],i), s, 'rx')
scatter(i.*ones(size(data_mat,1),1), data_mat_norm([1:size(data_mat,1)],i), s-5, 'b')
end
xticks([1:5]); xticklabels({des{kdx(1:end)}}); ylabel('Normalised value'); xlim([0,6])
legend('Synthetic data', 'Experimental data', 'linewidth', 2, 'location', 'southeast')
set(gca, 'Fontsize', f)

%% Tensile properties: 

des = {'Material_x', 'Cast_code', 'Temperature', 'PS02', 'UTS', 'Elongation',...
       'RA', 'C', 'Cr', 'Co', 'Al', 'Ti', 'Mo', 'W', 'Nb', 'Fe', 'B', 'V', 'Ni',...
       'Si', 'Mn', 'Cu', 'S', 'P', 'N', 'NbTa'};
   
% Identify the number of data that falls into the corresponding category of cast_code:
index = zeros(max(tensile_mat(:,2)),1); % Create empty array to record number of data for a cast_code category
for i = 1:max(tensile_mat(:,2))
idx = find(tensile_mat(:,2) == i); 
index(i) = length(idx);
end
% Identify the category of cast_code with the lowest and highest data-set:
fdx_min = find(index == min(index)); fdx_max = find(index == max(index));

% Find all rows corresponding to cast_code with lowest data-set: 
idx = find(tensile_mat(:,2) == fdx_min(1)); 
data = table1b(idx,:); data_mat = table2array(data);

% Define the correlation matrix of all features in the chosen category:
kdx = [3:7]; 
cormat = corrcoef(data_mat(:,kdx)); 
cormat_table = array2table(cormat, 'VariableNames', {des{kdx}}, 'RowNames', {des{kdx}});

figure; 
f = 18;
imagesc(cormat); colorbar;
set(gca,'YTick', [1:length(kdx)],'YTickLabel',{des{kdx}}, 'Fontsize', f);
set(gca,'XTick', [1:length(kdx)],'XTickLabel',{des{kdx}}, 'Fontsize', f);

% Enhance the data-set:
sel = sort(unique(data_mat(:,3)),'descend'); % Find unique values of the Temperature data
N = [1250*ones(1,8)];
for i = 1:length(sel)
jdx = find(data_mat(:,3) == sel(i));
syn_nom =  enhance(data_mat(jdx(1), kdx), N(i), cormat, pc); % Synthetic data matrix
if i==1
syn_mat = syn_nom;
else
syn_mat = [syn_mat; syn_nom];
end
end

% Create table of synthetic data:
syn_data = array2table([repmat(data_mat(1,[1:2]),Ndata,1), syn_mat, repmat(data_mat(1,[8:end]),Ndata,1)],...
           'VariableNames', des);
writetable(syn_data, 'Tensile_syndata_small.csv') 

% Plot the category of key input feature values:
figure;
for i = 1:length(kdx)
subplot(2,3,i)
hold on; box on; grid on;
f = 15; s = 18;
scatter(kdx(i).*ones(Ndata,1), syn_mat(:,i), s, 'rx')
scatter(kdx(i).*ones(length(idx),1), data_mat(:,kdx(i)), s+3, 'b')
xlabel(des{kdx(i)}); ylabel('Feature value');
set(gca, 'Fontsize', f); set(gca,'XTickLabel',[]);
end
legend('Synthetic data', 'Real data', 'linewidth', 2)

% Compute parameters for Violinplot:
data_mat_norm = normalize([data_mat(:,kdx); syn_mat]); % Normalize the data
for i=1:length(kdx)
[f, u, bb] = ksdensity(data_mat_norm([size(data_mat,1)+1:end],i));
f=f/max(f)*0.1; % Normalize the PDF
F(:,i)=f; U(:,i)=u; bw(:,i)=bb;
end

% Plot the Violinplot:
figure;
f = 18; s = 18;
hold on; box on; grid on;
for i = 1:length(kdx)
h = fill([F(:,i)+i;flipud(i-F(:,i))],[U(:,i);flipud(U(:,i))],'m','FaceAlpha',0.1,'EdgeColor','m', 'handlevisibility', 'off');
scatter(i.*ones(size(syn_mat,1),1), data_mat_norm([size(data_mat,1)+1:end],i), s, 'rx')
scatter(i.*ones(size(data_mat,1),1), data_mat_norm([1:size(data_mat,1)],i), s-5, 'b')
end
xticks([1:5]); xticklabels({des{kdx(1:end)}}); ylabel('Normalised value'); xlim([0,6])
legend('Synthetic data', 'Experimental data', 'linewidth', 2, 'location', 'southeast')
set(gca, 'Fontsize', f)
