
%% Football Analytics: Model A/B/C and Model B Modes
% Figure 1: Actual test metrics from Python output CSV
% Figure 2: Conceptual structure of Model B
%
% Run 02_train_models_ABC.py first so the results CSV exists.

clear; clc; close all;

%% 1. Find the real model-comparison CSV

candidateFiles = {
    fullfile(pwd, 'model_ABC_results', ...
        'ABC_model_comparison.csv')
    fullfile('C:\Users\rsuba\Downloads\R football analytics', ...
        'model_ABC_results', 'ABC_model_comparison.csv')
};

resultsFile = '';

for k = 1:numel(candidateFiles)
    if isfile(candidateFiles{k})
        resultsFile = candidateFiles{k};
        break;
    end
end

if isempty(resultsFile)
    error(['Could not find ABC_model_comparison.csv. ', ...
        'Edit candidateFiles to point to the actual CSV location.']);
end

fprintf('Reading actual results from:\n%s\n\n', resultsFile);

T = readtable(resultsFile);

% Confirm the expected model rows and columns exist
requiredColumns = {
    'model', 'test_MAE', 'test_RMSE', 'test_R2'
};

if ~all(ismember(requiredColumns, T.Properties.VariableNames))
    error('The CSV does not contain the expected test metric columns.');
end

% Ensure A, B, C appear in the intended order
modelNames = string(T.model);
[foundA, idxA] = ismember("A", modelNames);
[foundB, idxB] = ismember("B", modelNames);
[foundC, idxC] = ismember("C", modelNames);

if ~(foundA && foundB && foundC)
    error('Expected model rows A, B, and C were not found in the CSV.');
end

idx = [idxA idxB idxC];
T = T(idx, :);

modelLabels = {'Model A', 'Model B', 'Model C'};

%% 2. FIGURE 1 — Compare the three models using actual test metrics

MAE  = T.test_MAE;
RMSE = T.test_RMSE;
R2   = T.test_R2;

figure('Name', 'Model A vs B vs C', ...
    'Color', 'w', 'Position', [100 100 1200 650]);

tiledlayout(1, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

% MAE: lower is better
nexttile;
b = bar(MAE);
b.FaceColor = [0.20 0.48 0.72];
set(gca, 'XTick', 1:3, 'XTickLabel', modelLabels);
ylabel('Test MAE');
title('Mean Absolute Error');
grid on;
xtickangle(15);
for i = 1:3
    text(i, MAE(i), sprintf('%.4f', MAE(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end

% RMSE: lower is better
nexttile;
b = bar(RMSE);
b.FaceColor = [0.20 0.62 0.48];
set(gca, 'XTick', 1:3, 'XTickLabel', modelLabels);
ylabel('Test RMSE');
title('Root Mean Squared Error');
grid on;
xtickangle(15);
for i = 1:3
    text(i, RMSE(i), sprintf('%.4f', RMSE(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end

% R²: higher is better
nexttile;
b = bar(R2);
b.FaceColor = [0.85 0.55 0.22];
set(gca, 'XTick', 1:3, 'XTickLabel', modelLabels);
ylabel('Test R^2');
title('Explained Variance');
grid on;
xtickangle(15);
for i = 1:3
    text(i, R2(i), sprintf('%.4f', R2(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end

sgtitle('Football Analytics: Model A vs Model B vs Model C', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Display the exact values used in the chart
disp('ACTUAL TEST METRICS FROM CSV');
disp(table(modelLabels', MAE, RMSE, R2, ...
    'VariableNames', {'Model', 'Test_MAE', 'Test_RMSE', 'Test_R2'}));

exportgraphics(gcf, 'Figure1_Model_ABC_Comparison.png', ...
    'Resolution', 300);

%% 3. FIGURE 2 — Explain the three modes inside Model B
% These are components of one model, not three competing models.

figure('Name', 'Inside Model B', ...
    'Color', 'w', 'Position', [100 100 1200 650]);

axis off;
xlim([0 12]);
ylim([0 10]);
hold on;

% Title
text(6, 9.35, 'Inside Model B: Three Components', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 18, 'FontWeight', 'bold');

% MODE 1: target
rectangle('Position', [0.5 5.3 3.2 2.5], ...
    'Curvature', 0.08, 'FaceColor', [0.88 0.93 0.99], ...
    'EdgeColor', [0.20 0.40 0.70], 'LineWidth', 2);

text(2.1, 7.25, 'MODE 1', 'HorizontalAlignment', 'center', ...
    'FontSize', 15, 'FontWeight', 'bold');
text(2.1, 6.55, 'TARGET (Y)', 'HorizontalAlignment', 'center', ...
    'FontSize', 12, 'FontWeight', 'bold');
text(2.1, 5.95, 'post\_GA90', 'HorizontalAlignment', 'center', ...
    'FontSize', 13);
text(2.1, 5.55, 'What we predict', 'HorizontalAlignment', 'center', ...
    'FontSize', 10);

% MODE 2: algorithm
rectangle('Position', [4.4 5.3 3.2 2.5], ...
    'Curvature', 0.08, 'FaceColor', [0.88 0.96 0.89], ...
    'EdgeColor', [0.20 0.55 0.32], 'LineWidth', 2);

text(6, 7.25, 'MODE 2', 'HorizontalAlignment', 'center', ...
    'FontSize', 15, 'FontWeight', 'bold');
text(6, 6.55, 'PREDICTION METHOD', ...
    'HorizontalAlignment', 'center', 'FontSize', 11, ...
    'FontWeight', 'bold');
text(6, 6.05, 'Random Forest', ...
    'HorizontalAlignment', 'center', 'FontSize', 13);
text(6, 5.55, 'How we predict', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

% MODE 3: features
rectangle('Position', [8.3 5.3 3.2 2.5], ...
    'Curvature', 0.08, 'FaceColor', [1.00 0.94 0.84], ...
    'EdgeColor', [0.78 0.48 0.12], 'LineWidth', 2);

text(9.9, 7.25, 'MODE 3', 'HorizontalAlignment', 'center', ...
    'FontSize', 15, 'FontWeight', 'bold');
text(9.9, 6.65, 'INPUT FEATURES', 'HorizontalAlignment', 'center', ...
    'FontSize', 12, 'FontWeight', 'bold');
text(9.9, 6.15, 'Player + team data', ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
text(9.9, 5.65, 'What the model uses', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

% Clarify the input contents below the boxes
text(6, 4.35, ...
    'PLAYER FEATURES + SIX DESTINATION-TEAM PERFORMANCE FEATURES', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 11, 'FontWeight', 'bold');

text(6, 3.75, ...
    'Age, position, pre-transfer statistics, values and transfer fee', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

text(6, 3.30, ...
    'Team goals for/against, goal difference, win rate, points per game, match count', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

% Bottom summary
rectangle('Position', [2.0 1.0 8.0 1.35], ...
    'Curvature', 0.08, 'FaceColor', [0.94 0.94 0.94], ...
    'EdgeColor', [0.35 0.35 0.35], 'LineWidth', 1.5);

text(6, 1.85, 'MODEL B OUTPUT', 'HorizontalAlignment', 'center', ...
    'FontSize', 12, 'FontWeight', 'bold');

text(6, 1.35, ...
    'Predicted post-transfer goals + assists per 90 minutes', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

text(6, 0.45, ...
    'Mode 1 = what | Mode 2 = how | Mode 3 = which inputs', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

exportgraphics(gcf, 'Figure2_ModelB_Modes.png', ...
    'Resolution', 300);

fprintf('\nFinished. Saved figures:\n');
fprintf('  Figure1_Model_ABC_Comparison.png\n');
fprintf('  Figure2_ModelB_Modes.png\n');
