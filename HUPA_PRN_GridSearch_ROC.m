%% HUPA_PRN_GridSearch_ROC.m
% Purpose:
%   Performs a Grid Search with Cross-Validation (CV) to optimize AUC,
%   followed by a final evaluation on a hold-out Test set for various models.
%   The analysis is performed on specific AVCA + Complexity feature groups.
%
%   This version runs the full pipeline on TWO datasets:
%     - HUPA_voice_features_PRN_CPP_50kHz.csv
%     - HUPA_voice_features_PRN_CPP_44_1kHz.csv
%
% Workflow (per dataset):
%   1. Load data.
%   2. Group features (Noise, Perturbation, Tremor, Complexity).
%   3. For each group:
%       a. Clean data (remove constant cols, impute NaNs).
%       b. Split into Train (80%) and Test (20%).
%       c. Train models using 5-fold CV on the Train set to find best hyperparameters.
%       d. Retrain best model on full Train set and evaluate on Test set.
%   4. Summarize results and plot ROC curves.
%
% Requirements:
%   - Statistics and Machine Learning Toolbox (fitclinear, fitcsvm, TreeBagger, fitcknn, etc.)
%   - Optional: fitcnet (for MLP Neural Network). If missing, MLP is skipped.

clear; clc; close all;

%% ======================== 1) PATHS AND DATASETS =========================
% Detect current path
currentPath = fileparts(mfilename('fullpath'));

% Base data folder
dataDir = fullfile(currentPath, 'data');

% List of CSV files and labels (per sampling rate)
csvFiles = { ...
    'HUPA_voice_features_PRN_CPP_50kHz.csv', ...
    'HUPA_voice_features_PRN_CPP_44_1kHz.csv' ...
};

fsLabels = { ...
    '50 kHz', ...
    '44.1 kHz' ...
};

% Check if fitcnet (Neural Network) is available in this version
hasFitcnet = ~isempty(which('fitcnet'));

%% ======================== 2) LOOP OVER DATASETS =========================
for ds = 1:numel(csvFiles)

    csvPath = fullfile(dataDir, csvFiles{ds});
    fsLabel = fsLabels{ds};

    fprintf('\n=========================================================\n');
    fprintf('   Running Grid Search + ROC for dataset: %s (%s)\n', ...
        csvFiles{ds}, fsLabel);
    fprintf('=========================================================\n');

    if ~exist(csvPath, 'file')
        warning('File not found: %s. Skipping this dataset.\n', csvPath);
        continue;
    end

    %% ======================== 3) LOADING DATA ===========================
    T = readtable(csvPath);

    % Ensure the target variable exists
    if ~ismember('Label', T.Properties.VariableNames)
        error('Column "Label" not found in the CSV: %s', csvPath);
    end

    y = T.Label;
    if ~isnumeric(y)
        y = double(y);
    end

    %% ================= 4) DEFINITION OF FEATURE GROUPS ==================
    % --- Noise Features ---
    noiseCols = {'HNR_mean','HNR_std', ...
                 'CHNR_mean','CHNR_std', ...
                 'GNE_mean','GNE_std', ...
                 'NNE_mean','NNE_std'};

    % --- Perturbation Features (CPP + jitter/shimmer) ---
    perturbCols = { ...
        'CPP', ...        % Cepstral Peak Prominence
        'rShimmer', ...   % Relative Shimmer
        'rJitta','rJitt','rRrRAP','rPPQ','rSPPQ', ...
        'rShdB','rAPQ','rSAPQ'};

    % --- Tremor Features ---
    tremorCols = {'rFTRI','rATRI','rFftr','rFatr'};

    % --- Complexity Features ---
    complexCols = { ...
        'rApEn_mean','rApEn_std', ...
        'rSampEn_mean','rSampEn_std', ...
        'rFuzzyEn_mean','rFuzzyEn_std', ...
        'rGSampEn_mean','rGSampEn_std', ...
        'rmSampEn_mean','rmSampEn_std', ...
        'CorrDim_mean','CorrDim_std', ...
        'LLE_mean','LLE_std', ...
        'Hurst_mean','Hurst_std', ...
        'mDFA_mean','mDFA_std', ...
        'RPDE_mean','RPDE_std', ...
        'PE_mean','PE_std', ...
        'MarkEnt_mean','MarkEnt_std'};

    % Ensure we only use columns that actually exist in the loaded table
    noiseCols      = intersect(noiseCols,      T.Properties.VariableNames, 'stable');
    perturbCols    = intersect(perturbCols,    T.Properties.VariableNames, 'stable');
    tremorCols     = intersect(tremorCols,     T.Properties.VariableNames, 'stable');
    complexCols    = intersect(complexCols,    T.Properties.VariableNames, 'stable');

    % Create a structure to iterate over groups later
    groups = struct();
    groups.noise        = noiseCols;
    groups.perturbation = perturbCols;
    groups.tremor       = tremorCols;
    groups.complexity   = complexCols;
    % Optional: A group containing 'all' features combined
    groups.all          = unique([noiseCols, perturbCols, tremorCols, complexCols]);

    groupNames = fieldnames(groups);
    groupOrderForPlots = {'noise','perturbation','tremor','complexity'};

    fprintf('Feature groups defined for %s:\n', fsLabel);
    for gi = 1:numel(groupNames)
        g = groupNames{gi};
        fprintf('  %-12s : %2d features\n', g, numel(groups.(g)));
    end

    %% ================= 5) MAIN LOOP: GRID SEARCH + CV + TEST ============
    rng(42); % Set seed for reproducibility (per dataset)

    summaryRows = {};  % To store: {Group, Model, CvAUC, TestAUC}
    rocStruct   = struct();

    for gi = 1:numel(groupNames)
        groupName = groupNames{gi};
        featCols  = groups.(groupName);

        if isempty(featCols)
            warning('Group "%s" has no available features in %s. Skipping.', ...
                groupName, fsLabel);
            continue;
        end

        % ================= PREPARE X AND Y ===================================
        X    = table2array(T(:, featCols));
        yVec = y;

        % 0) Treat Infinite values as NaN (to be imputed later)
        X(~isfinite(X)) = NaN;

        % 1) Detect "broken" columns:
        %    - All NaN
        %    - Constant values (variance = 0, ignoring NaNs)
        allNaN   = all(isnan(X), 1);
        allConst = false(1, size(X,2));

        for j = 1:size(X,2)
            col   = X(:, j);
            colNN = col(~isnan(col));   % Remove NaNs for check
            if numel(colNN) <= 1
                allConst(j) = true;
            else
                if std(colNN) == 0
                    allConst(j) = true;
                end
            end
        end

        badCols = allNaN | allConst;

        if any(badCols)
            fprintf('  [%s - %s] Removed columns (NaN or Constant):\n', ...
                fsLabel, groupName);
            disp(featCols(badCols));
        end

        X       = X(:, ~badCols);
        featUse = featCols(~badCols); %#ok<NASGU>

        % If no valid columns remain, skip group
        if isempty(X)
            warning('Group "%s" (%s): no valid features after NaN/constant removal. Skipping.', ...
                groupName, fsLabel);
            continue;
        end

        % 2) Impute remaining NaNs using the Median of each column
        for j = 1:size(X,2)
            col   = X(:, j);
            maskN = isnan(col);
            if any(maskN)
                med = median(col(~maskN));
                if isempty(med) || isnan(med)
                    med = 0;  % Pathological case fallback
                end
                col(maskN) = med;
                X(:, j)    = col;
            end
        end

        % 3) Check Class Balance
        if any(~isfinite(yVec))
            error('y (Label) contains non-finite values. Please check CSV: %s', csvPath);
        end

        if numel(unique(yVec)) < 2
            warning('Group "%s" (%s): only one class found in labels. Skipping.', ...
                groupName, fsLabel);
            continue;
        end

        % ================= SPLIT TRAIN / TEST ================================
        % Outer split: 80% Train (for Grid Search CV), 20% Test (for final Eval)
        cvOuter = cvpartition(yVec, 'HoldOut', 0.20);
        X_train = X(training(cvOuter), :);
        y_train = yVec(training(cvOuter));
        X_test  = X(test(cvOuter), :);
        y_test  = yVec(test(cvOuter));

        fprintf('\n=== [%s] Group: %s (%d features, %d train / %d test) ===\n', ...
            fsLabel, groupName, size(X_train,2), numel(y_train), numel(y_test));

        % ---------- Model 1: Logistic Regression ----------
        try
            [cvAUC, testAUC, fpr, tpr] = model_logreg(X_train, y_train, X_test, y_test);
            fprintf('  Logistic Regression: CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'logreg', cvAUC, testAUC}; %#ok<AGROW>
            rocStruct.(groupName).logreg.fpr      = fpr;
            rocStruct.(groupName).logreg.tpr      = tpr;
            rocStruct.(groupName).logreg.testAUC  = testAUC;
        catch ME
            warning('  Logistic Regression failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 2: SVM RBF ----------
        try
            [cvAUC, testAUC, fpr, tpr] = model_svm_rbf(X_train, y_train, X_test, y_test);
            fprintf('  SVM RBF:            CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'svm_rbf', cvAUC, testAUC};
            rocStruct.(groupName).svm_rbf.fpr     = fpr;
            rocStruct.(groupName).svm_rbf.tpr     = tpr;
            rocStruct.(groupName).svm_rbf.testAUC = testAUC;
        catch ME
            warning('  SVM RBF failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 3: Random Forest (TreeBagger) ----------
        try
            [cvAUC, testAUC, fpr, tpr] = model_random_forest(X_train, y_train, X_test, y_test);
            fprintf('  Random Forest:      CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'rf', cvAUC, testAUC};
            rocStruct.(groupName).rf.fpr          = fpr;
            rocStruct.(groupName).rf.tpr          = tpr;
            rocStruct.(groupName).rf.testAUC      = testAUC;
        catch ME
            warning('  Random Forest failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 4: k-NN ----------
        try
            [cvAUC, testAUC, fpr, tpr] = model_knn(X_train, y_train, X_test, y_test);
            fprintf('  k-NN:               CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'knn', cvAUC, testAUC};
            rocStruct.(groupName).knn.fpr         = fpr;
            rocStruct.(groupName).knn.tpr         = tpr;
            rocStruct.(groupName).knn.testAUC     = testAUC;
        catch ME
            warning('  k-NN failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 5: Neural Network (fitcnet) ----------
        if hasFitcnet
            try
                [cvAUC, testAUC, fpr, tpr] = model_mlp(X_train, y_train, X_test, y_test);
                fprintf('  Neural Network:     CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
                summaryRows(end+1,:) = {groupName, 'mlp', cvAUC, testAUC};
                rocStruct.(groupName).mlp.fpr      = fpr;
                rocStruct.(groupName).mlp.tpr      = tpr;
                rocStruct.(groupName).mlp.testAUC  = testAUC;
            catch ME
                warning('  Neural Network (fitcnet) failed for group "%s" (%s): %s', ...
                    groupName, fsLabel, ME.message);
            end
        else
            fprintf('  Neural Network:     skipped (fitcnet not available).\n');
        end
    end

    %% ===================== 6) RESULTS SUMMARY TABLE =====================
    if ~isempty(summaryRows)
        summaryTable = cell2table(summaryRows, ...
            'VariableNames', {'Group','Model','CvAUC','TestAUC'});
        fprintf('\n=========== Overall summary for %s (sorted by Group, Test AUC) ===========\n', ...
            fsLabel);
        % Sort by Group Name (ascending) then Test AUC (descending)
        summaryTable = sortrows(summaryTable, {'Group','TestAUC'}, {'ascend','descend'});
        disp(summaryTable);
    else
        warning('No successful model runs to summarize for dataset %s.', fsLabel);
    end

    %% ====================== 7) ROC PLOTS (4 SUBPLOTS) =======================
    modelOrder = {'logreg','svm_rbf','rf','knn','mlp'};
    prettyName.logreg  = 'Logistic';
    prettyName.svm_rbf = 'SVM RBF';
    prettyName.rf      = 'Random Forest';
    prettyName.knn     = 'k-NN';
    prettyName.mlp     = 'Neural Net';
    
    f = figure;
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    
    for gi = 1:numel(groupOrderForPlots)
        gname = groupOrderForPlots{gi};
        if ~isfield(rocStruct, gname)
            continue;
        end
        nexttile;
        hold on;
        for mi = 1:numel(modelOrder)
            mname = modelOrder{mi};
            if isfield(rocStruct.(gname), mname)
                fpr = rocStruct.(gname).(mname).fpr;
                tpr = rocStruct.(gname).(mname).tpr;
                auc = rocStruct.(gname).(mname).testAUC;
                plot(fpr, tpr, 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('%s (AUC=%.2f)', prettyName.(mname), auc));
            end
        end
        plot([0 1],[0 1],'k--','LineWidth',1);
        axis square;
        xlim([0 1]); ylim([0 1]);
        xlabel('False positive rate');
        ylabel('True positive rate');
        switch gname
            case 'noise',        title('Noise features');
            case 'perturbation', title('Perturbation features');
            case 'tremor',       title('Tremor features');
            case 'complexity',   title('Complexity features');
            otherwise,           title(gname);
        end
        legend('Location','SouthEast','Box','off');
        hold off;
    end
    
    % ==== SAVE FIG ====
    if contains(csvPath, '50kHz')
        fs_suffix = '50kHz';
    elseif contains(csvPath, '44_1kHz')
        fs_suffix = '44_1kHz';
    else
        fs_suffix = 'unknownFS';
    end
    
    fig_dir = fullfile(currentPath, 'figures');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end
    
    tool_suffix = 'MATLAB';  
    
    file_base = sprintf('ROC_HUPA_%s_%s', fs_suffix, tool_suffix);
    % Example:
    %   ROC_HUPA_50kHz_MATLAB
    %   ROC_HUPA_44_1kHz_MATLAB
    
    png_path = fullfile(fig_dir, [file_base '.png']);
    pdf_path = fullfile(fig_dir, [file_base '.pdf']);
    
    saveas(f, png_path);
    saveas(f, pdf_path);
    
    fprintf('\nROC figure saved to:\n  %s\n  %s\n', png_path, pdf_path);

end % end loop over datasets

%% ===================== 8) MODEL SUBFUNCTIONS ============================

function [cvAUC, testAUC, fprTest, tprTest] = model_logreg(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Logistic Regression using 'fitclinear'.
% Tunes: Lambda (Regularization strength) and Regularization type (Ridge/Lasso).

cvInner = cvpartition(ytrain,'KFold',5);
lambdaGrid = logspace(-4,1,6);    % [1e-4 ... 10]
regGrid    = {'ridge','lasso'};   % L2 and L1

bestCvAUC = -inf;
bestReg   = '';
bestLambda = NaN;

% --- Grid Search ---
for ri = 1:numel(regGrid)
    regType = regGrid{ri};
    for li = 1:numel(lambdaGrid)
        lam = lambdaGrid(li);
        foldAUC = zeros(cvInner.NumTestSets,1);
        for k = 1:cvInner.NumTestSets
            idxTr  = training(cvInner,k);
            idxVal = test(cvInner,k);
            
            % Standardize inside the loop (prevent data leakage)
            [Xtr, mu, sigma] = zscore(Xtrain(idxTr,:));
            Xval = (Xtrain(idxVal,:) - mu) ./ sigma;

            mdl = fitclinear(Xtr, ytrain(idxTr), ...
                'Learner','logistic', ...
                'Regularization', regType, ...
                'Lambda', lam, ...
                'ClassNames',[0 1]);

            [~, scoreVal] = predict(mdl, Xval);
            scores = scoreVal(:,2); % Positive class scores
            [~,~,~,aucVal] = perfcurve(ytrain(idxVal), scores, 1);
            foldAUC(k) = aucVal;
        end
        meanAUC = mean(foldAUC);
        if meanAUC > bestCvAUC
            bestCvAUC = meanAUC;
            bestReg   = regType;
            bestLambda = lam;
        end
    end
end

% --- Retrain Best Model on Full Train Set ---
[XtrAll, muAll, sigmaAll] = zscore(Xtrain);
mdlBest = fitclinear(XtrAll, ytrain, ...
    'Learner','logistic', ...
    'Regularization', bestReg, ...
    'Lambda', bestLambda, ...
    'ClassNames',[0 1]);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(mdlBest, XtestZ);
scoresTest = scoreTest(:,2);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);
cvAUC = bestCvAUC;
end

% -------------------------------------------------------------------------
function [cvAUC, testAUC, fprTest, tprTest] = model_svm_rbf(Xtrain, ytrain, Xtest, ytest)
% Grid Search for SVM with RBF Kernel.
% Tunes: BoxConstraint (C) and KernelScale.
% Uses 'fitPosterior' to get probability estimates for AUC.

cvInner   = cvpartition(ytrain,'KFold',5);
Cgrid     = [0.1 0.3 1 3 10 30];
scaleGrid = [0.1 0.3 1 3 10];

bestCvAUC = -inf;
bestC     = NaN;
bestScale = NaN;

% --- Grid Search ---
for ci = 1:numel(Cgrid)
    C = Cgrid(ci);
    for si = 1:numel(scaleGrid)
        ks = scaleGrid(si);
        foldAUC = zeros(cvInner.NumTestSets,1);
        for k = 1:cvInner.NumTestSets
            idxTr  = training(cvInner,k);
            idxVal = test(cvInner,k);
            
            % Z-score normalization
            [Xtr, mu, sigma] = zscore(Xtrain(idxTr,:));
            Xval = (Xtrain(idxVal,:) - mu) ./ sigma;

            mdl = fitcsvm(Xtr, ytrain(idxTr), ...
                'KernelFunction','rbf', ...
                'KernelScale', ks, ...
                'BoxConstraint', C, ...
                'Standardize', false, ...
                'ClassNames',[0 1]);

            [~, scoreVal] = predict(mdl, Xval);
            scores = scoreVal(:,2);
            [~,~,~,aucVal] = perfcurve(ytrain(idxVal), scores, 1);
            foldAUC(k) = aucVal;
        end
        meanAUC = mean(foldAUC);
        if meanAUC > bestCvAUC
            bestCvAUC = meanAUC;
            bestC     = C;
            bestScale = ks;
        end
    end
end

% --- Retrain Best Model ---
[XtrAll, muAll, sigmaAll] = zscore(Xtrain);
mdlBest = fitcsvm(XtrAll, ytrain, ...
    'KernelFunction','rbf', ...
    'KernelScale', bestScale, ...
    'BoxConstraint', bestC, ...
    'Standardize', false, ...
    'ClassNames',[0 1]);

% Fit posterior to transform scores to probabilities
mdlBest = fitPosterior(mdlBest, XtrAll, ytrain);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(mdlBest, XtestZ);
scoresTest = scoreTest(:,2);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);
cvAUC = bestCvAUC;
end

% -------------------------------------------------------------------------
function [cvAUC, testAUC, fprTest, tprTest] = model_random_forest(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Random Forest using 'TreeBagger'.
% Tunes: Number of Trees and MinLeafSize.

cvInner   = cvpartition(ytrain,'KFold',5);
nTreeGrid = [200 400 800];
leafGrid  = [1 2 5];

bestCvAUC = -inf;
bestTrees = NaN;
bestLeaf  = NaN;

% --- Grid Search ---
for ti = 1:numel(nTreeGrid)
    nTrees = nTreeGrid(ti);
    for li = 1:numel(leafGrid)
        leaf = leafGrid(li);
        foldAUC = zeros(cvInner.NumTestSets,1);
        for k = 1:cvInner.NumTestSets
            idxTr  = training(cvInner,k);
            idxVal = test(cvInner,k);

            mdl = TreeBagger(nTrees, Xtrain(idxTr,:), categorical(ytrain(idxTr)), ...
                'Method','classification', ...
                'MinLeafSize', leaf, ...
                'OOBPrediction','off');

            [~, scoreVal] = predict(mdl, Xtrain(idxVal,:));
            
            % Handle categorical class names safely
            classNames = mdl.ClassNames;
            if isa(classNames,'categorical')
                posIdx = find(classNames == categorical(1));
            else
                posIdx = find(strcmp(classNames, '1'));
            end
            if isempty(posIdx)
                posIdx = 2; % Fallback binary
            end
            scores = scoreVal(:,posIdx);
            [~,~,~,aucVal] = perfcurve(ytrain(idxVal), scores, 1);
            foldAUC(k) = aucVal;
        end
        meanAUC = mean(foldAUC);
        if meanAUC > bestCvAUC
            bestCvAUC = meanAUC;
            bestTrees = nTrees;
            bestLeaf  = leaf;
        end
    end
end

% --- Retrain Best Model ---
mdlBest = TreeBagger(bestTrees, Xtrain, categorical(ytrain), ...
    'Method','classification', ...
    'MinLeafSize', bestLeaf, ...
    'OOBPrediction','off');

% --- Evaluate on Test Set ---
[~, scoreTest] = predict(mdlBest, Xtest);
classNames = mdlBest.ClassNames;
if isa(classNames,'categorical')
    posIdx = find(classNames == categorical(1));
else
    posIdx = find(strcmp(classNames, '1'));
end
if isempty(posIdx)
    posIdx = 2;
end
scoresTest = scoreTest(:,posIdx);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);
cvAUC = bestCvAUC;
end

% -------------------------------------------------------------------------
function [cvAUC, testAUC, fprTest, tprTest] = model_knn(Xtrain, ytrain, Xtest, ytest)
% Grid Search for k-Nearest Neighbors (k-NN).
% Tunes: Number of Neighbors (k) and Distance Metric.
% Performs manual Z-score standardization.

cvInner = cvpartition(ytrain,'KFold',5);
kGrid   = [3 5 7 9 11];
distGrid = {'euclidean','cityblock'};

bestCvAUC = -inf;
bestK     = NaN;
bestDist  = '';

% --- Grid Search ---
for ki = 1:numel(kGrid)
    kVal = kGrid(ki);
    for di = 1:numel(distGrid)
        dist = distGrid{di};
        foldAUC = zeros(cvInner.NumTestSets,1);
        for k = 1:cvInner.NumTestSets
            idxTr  = training(cvInner,k);
            idxVal = test(cvInner,k);
            
            [Xtr, mu, sigma] = zscore(Xtrain(idxTr,:));
            Xval = (Xtrain(idxVal,:) - mu) ./ sigma;

            mdl = fitcknn(Xtr, ytrain(idxTr), ...
                'NumNeighbors', kVal, ...
                'Distance', dist, ...
                'Standardize', false, ...
                'ClassNames',[0 1]);

            [~, scoreVal] = predict(mdl, Xval);
            scores = scoreVal(:,2);
            [~,~,~,aucVal] = perfcurve(ytrain(idxVal), scores, 1);
            foldAUC(k) = aucVal;
        end
        meanAUC = mean(foldAUC);
        if meanAUC > bestCvAUC
            bestCvAUC = meanAUC;
            bestK     = kVal;
            bestDist  = dist;
        end
    end
end

% --- Retrain Best Model ---
[XtrAll, muAll, sigmaAll] = zscore(Xtrain);
mdlBest = fitcknn(XtrAll, ytrain, ...
    'NumNeighbors', bestK, ...
    'Distance', bestDist, ...
    'Standardize', false, ...
    'ClassNames',[0 1]);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(mdlBest, XtestZ);
scoresTest = scoreTest(:,2);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);
cvAUC = bestCvAUC;
end

% -------------------------------------------------------------------------
function [cvAUC, testAUC, fprTest, tprTest] = model_mlp(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Shallow Neural Network (MLP) using 'fitcnet'.
% Tunes: Layer Sizes and Lambda (Regularization).

if isempty(which('fitcnet'))
    error('fitcnet not found in this MATLAB version.');
end

cvInner    = cvpartition(ytrain,'KFold',5);
layerGrid  = {32, 64, [64 32], [128 64]};
lambdaGrid = [1e-4 1e-3 1e-2];

bestCvAUC  = -inf;
bestLayers = [];
bestLambda = NaN;

% --- Grid Search ---
for li = 1:numel(layerGrid)
    layers = layerGrid{li};
    for lj = 1:numel(lambdaGrid)
        lam = lambdaGrid(lj);
        foldAUC = zeros(cvInner.NumTestSets,1);
        for k = 1:cvInner.NumTestSets
            idxTr  = training(cvInner,k);
            idxVal = test(cvInner,k);
            
            [Xtr, mu, sigma] = zscore(Xtrain(idxTr,:));
            Xval = (Xtrain(idxVal,:) - mu) ./ sigma;

            Mdl = fitcnet(Xtr, ytrain(idxTr), ...
                'Standardize', false, ...
                'LayerSizes', layers, ...
                'Lambda', lam, ...
                'IterationLimit', 1000, ...
                'Verbose', 0, ...
                'ClassNames',[0 1]);

            [~, scoreVal] = predict(Mdl, Xval);
            classNames = Mdl.ClassNames;
            if isnumeric(classNames)
                posIdx = find(classNames == 1);
            else
                posIdx = find(strcmp(string(classNames), '1'));
            end
            if isempty(posIdx)
                posIdx = 2;
            end
            scores = scoreVal(:,posIdx);
            [~,~,~,aucVal] = perfcurve(ytrain(idxVal), scores, 1);
            foldAUC(k) = aucVal;
        end
        meanAUC = mean(foldAUC);
        if meanAUC > bestCvAUC
            bestCvAUC  = meanAUC;
            bestLayers = layers;
            bestLambda = lam;
        end
    end
end

% --- Retrain Best Model ---
[XtrAll, muAll, sigmaAll] = zscore(Xtrain);
MdlBest = fitcnet(XtrAll, ytrain, ...
    'Standardize', false, ...
    'LayerSizes', bestLayers, ...
    'Lambda', bestLambda, ...
    'IterationLimit', 1000, ...
    'Verbose', 0, ...
    'ClassNames',[0 1]);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(MdlBest, XtestZ);
classNames = MdlBest.ClassNames;
if isnumeric(classNames)
    posIdx = find(classNames == 1);
else
    posIdx = find(strcmp(string(classNames), '1'));
end
if isempty(posIdx)
    posIdx = 2;
end
scoresTest = scoreTest(:,posIdx);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);
cvAUC = bestCvAUC;
end