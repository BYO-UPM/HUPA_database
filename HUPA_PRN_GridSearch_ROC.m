%% HUPA_PRN_GridSearch_ROC.m
% Purpose:
%   Performs a Grid Search with Cross-Validation (CV) to optimize AUC,
%   followed by a final evaluation on a hold-out Test set for various models.
%   The analysis is performed on specific AVCA + Complexity feature groups.
%
%   This version runs the full pipeline on TWO datasets:
%     - HUPA_voice_features_PRN_CPP_50kHz.csv
%     - HUPA_voice_features_PRN_CPP_25kHz.csv
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
% Prefer CSVs with metadata (Sex, Pathology code). Fallback to legacy names.
csvFilesPreferred = { ...
    'HUPA_voice_features_PRN_CPP_50kHz_with_meta.csv', ...
    'HUPA_voice_features_PRN_CPP_25kHz_with_meta.csv' ...
};

csvFilesFallback = { ...
    'HUPA_voice_features_PRN_CPP_50kHz.csv', ...
    'HUPA_voice_features_PRN_CPP_25kHz.csv' ...
};

fsLabels = { ...
    '50 kHz', ...
    '25 kHz' ...
};

% Check if fitcnet (Neural Network) is available in this version
hasFitcnet = ~isempty(which('fitcnet'));

%% ======================== 2) LOOP OVER DATASETS =========================
for ds = 1:numel(csvFilesPreferred)

    csvPathPreferred = fullfile(dataDir, csvFilesPreferred{ds});
    csvPathFallback  = fullfile(dataDir, csvFilesFallback{ds});

    if exist(csvPathPreferred, 'file')
        csvPath = csvPathPreferred;
    else
        csvPath = csvPathFallback;
    end
    fsLabel = fsLabels{ds};

    fprintf('\n=========================================================\n');
    fprintf('   Running Grid Search + ROC for dataset: %s (%s)\n', ...
        csvFilesPreferred{ds}, fsLabel);
    fprintf('=========================================================\n');

    if ~exist(csvPath, 'file')
        warning('File not found: %s. Skipping this dataset.\n', csvPath);
        continue;
    end

    %% ======================== 3) LOADING DATA ===========================
    T = readtable(csvPath);

    % Detect optional metadata columns (added by HUPA_Features_Extraction_v2.m)
    varNames = T.Properties.VariableNames;

    sexVar = '';
    if ismember('Sex', varNames)
        sexVar = 'Sex';
    end

    pathCodeVar = '';
    if ismember('Pathology_code', varNames)
        pathCodeVar = 'Pathology_code';
    elseif ismember('Pathologycode', varNames)
        pathCodeVar = 'Pathologycode';
    elseif ismember('Pathology_code_', varNames)
        pathCodeVar = 'Pathology_code_';
    end

    % Load optional pathology map (code -> name) from HUPA_db.xlsx
    pathologyMap = containers.Map('KeyType','int32','ValueType','char');
    xlsxPath = fullfile(dataDir, 'HUPA_db.xlsx');
    if exist(xlsxPath, 'file')
        try
            pathologyMap = load_pathology_map(xlsxPath);
            fprintf('Loaded pathology map from: %s (n=%d)\n', xlsxPath, pathologyMap.Count);
        catch ME
            warning('Could not load pathology map from %s: %s', xlsxPath, ME.message);
        end
    end


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
    groupOrderForPlots = {'noise','perturbation','tremor','complexity','all'};

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
        featUse = featCols(~badCols); 

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

        
        % Cache to store per-model test scores (for best-model analysis)
        evalCache = struct();

        % Optional metadata on the TEST set
        if ~isempty(sexVar)
            sexTest = string(T.(sexVar)(test(cvOuter)));
        else
            sexTest = [];
        end


if ~isempty(pathCodeVar)
    pathCodeTest = int32(T.(pathCodeVar)(test(cvOuter)));
    pathCodeAll  = int32(T.(pathCodeVar));
else
    pathCodeTest = [];
    pathCodeAll  = [];
end

if ~isempty(sexVar)
    sexAll = string(T.(sexVar));
else
    sexAll = [];
end

% ---------- Model 1: Logistic Regression ----------
        try
            [cvAUC, testAUC, fpr, tpr, scoresTest, thrYouden] = model_logreg(X_train, y_train, X_test, y_test);
            fprintf('  Logistic Regression: CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'logreg', cvAUC, testAUC}; 
            rocStruct.(groupName).logreg.fpr      = fpr;
            rocStruct.(groupName).logreg.tpr      = tpr;
            rocStruct.(groupName).logreg.testAUC  = testAUC;
            evalCache.logreg.scoresTest = scoresTest;
            evalCache.logreg.thrYouden   = thrYouden;
            evalCache.logreg.testAUC     = testAUC;
        catch ME
            warning('  Logistic Regression failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 2: SVM RBF ----------
        try
            [cvAUC, testAUC, fpr, tpr, scoresTest, thrYouden] = model_svm_rbf(X_train, y_train, X_test, y_test);
            fprintf('  SVM RBF:            CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'svm_rbf', cvAUC, testAUC};
            rocStruct.(groupName).svm_rbf.fpr     = fpr;
            rocStruct.(groupName).svm_rbf.tpr     = tpr;
            rocStruct.(groupName).svm_rbf.testAUC = testAUC;
            evalCache.svm_rbf.scoresTest = scoresTest;
            evalCache.svm_rbf.thrYouden   = thrYouden;
            evalCache.svm_rbf.testAUC     = testAUC;
        catch ME
            warning('  SVM RBF failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 3: Random Forest (TreeBagger) ----------
        try
            [cvAUC, testAUC, fpr, tpr, scoresTest, thrYouden] = model_random_forest(X_train, y_train, X_test, y_test);
            fprintf('  Random Forest:      CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
            summaryRows(end+1,:) = {groupName, 'rf', cvAUC, testAUC};
            rocStruct.(groupName).rf.fpr          = fpr;
            rocStruct.(groupName).rf.tpr          = tpr;
            rocStruct.(groupName).rf.testAUC      = testAUC;
            evalCache.rf.scoresTest = scoresTest;
            evalCache.rf.thrYouden   = thrYouden;
            evalCache.rf.testAUC     = testAUC;
        catch ME
            warning('  Random Forest failed for group "%s" (%s): %s', ...
                groupName, fsLabel, ME.message);
        end

        % ---------- Model 4: Neural Network (fitcnet) ----------
        if hasFitcnet
            try
                [cvAUC, testAUC, fpr, tpr, scoresTest, thrYouden] = model_mlp(X_train, y_train, X_test, y_test);
                fprintf('  Neural Network:     CV AUC = %.3f | Test AUC = %.3f\n', cvAUC, testAUC);
                summaryRows(end+1,:) = {groupName, 'mlp', cvAUC, testAUC};
                rocStruct.(groupName).mlp.fpr      = fpr;
                rocStruct.(groupName).mlp.tpr      = tpr;
                rocStruct.(groupName).mlp.testAUC  = testAUC;
                evalCache.mlp.scoresTest = scoresTest;
                evalCache.mlp.thrYouden   = thrYouden;
                evalCache.mlp.testAUC     = testAUC;
            catch ME
                warning('  Neural Network (fitcnet) failed for group "%s" (%s): %s', ...
                    groupName, fsLabel, ME.message);
            end
        else
            fprintf('  Neural Network:     skipped (fitcnet not available).\n');
        end

        % ---------------- Reviewer analyses: best model per group ----------------
        modelKeys = fieldnames(evalCache);
        if ~isempty(modelKeys)
            % Pick best by Test AUC
            bestKey = modelKeys{1};
            bestAUC = evalCache.(bestKey).testAUC;
            for kk = 2:numel(modelKeys)
                kname = modelKeys{kk};
                if evalCache.(kname).testAUC > bestAUC
                    bestAUC = evalCache.(kname).testAUC;
                    bestKey = kname;
                end
            end

            scoresBest = evalCache.(bestKey).scoresTest;
            thrBest    = evalCache.(bestKey).thrYouden;

            % Confusion matrix on TEST
            yPred = double(scoresBest >= thrBest);
            cm = confusionmat(y_test, yPred, 'Order', [0 1]);

            [sens, spec, bacc, acc] = confusion_metrics_from_cm(cm);

            % Save confusion matrix figure
            cmDir = fullfile(currentPath, 'figures', 'confusion_matrices');
            if ~exist(cmDir, 'dir'); mkdir(cmDir); end
            cmPath = fullfile(cmDir, sprintf('CM_%s_%s_%s.png', fsLabelToSuffix(fsLabel), groupName, bestKey));
            save_confusion_matrix_figure(cm, sprintf('%s - %s - %s', fsLabel, groupName, bestKey), cmPath);

            % AUC by Sex (TEST)
            aucBySexStr = "";
            if ~isempty(sexTest)
                uniqSex = unique(sexTest);
                parts = strings(numel(uniqSex),1);
                for si = 1:numel(uniqSex)
                    sx = uniqSex(si);
                    mask = (sexTest == sx);
                    if numel(unique(y_test(mask))) < 2
                        aucSx = NaN;
                    else
                        [~,~,~,aucSx] = perfcurve(y_test(mask), scoresBest(mask), 1);
                    end
                    parts(si) = sprintf('%s=%.3f', sx, aucSx);
                end
                aucBySexStr = strjoin(parts, '; ');
            end

            
% -----------------------------
% Subtype error audit (OOF full dataset)
% -----------------------------
subtypeAuditPath_OOF = "";
aucBySexOOFStr = "";

try
    % Out-of-fold scores for ALL samples (KFold=5)
    cvAudit = cvpartition(yVec, 'KFold', 5);
    scoresOOF = nan(size(yVec));

    for fk = 1:cvAudit.NumTestSets
        idxTr  = training(cvAudit, fk);
        idxVal = test(cvAudit, fk);

        XtrFold = X(idxTr, :);
        ytrFold = yVec(idxTr);
        XvalFold = X(idxVal, :);
        yvalFold = yVec(idxVal);

        switch bestKey
            case 'logreg'
                [~, ~, ~, ~, scoresVal, ~] = model_logreg(XtrFold, ytrFold, XvalFold, yvalFold);
            case 'svm_rbf'
                [~, ~, ~, ~, scoresVal, ~] = model_svm_rbf(XtrFold, ytrFold, XvalFold, yvalFold);
            case 'rf'
                [~, ~, ~, ~, scoresVal, ~] = model_random_forest(XtrFold, ytrFold, XvalFold, yvalFold);
            case 'mlp'
                [~, ~, ~, ~, scoresVal, ~] = model_mlp(XtrFold, ytrFold, XvalFold, yvalFold);
            otherwise
                scoresVal = nan(sum(idxVal),1);
        end
        scoresOOF(idxVal) = scoresVal;
    end

    % Global Youden threshold from OOF scores
    [fprO, tprO, thrO] = perfcurve(yVec, scoresOOF, 1);
    [~, idxThrO] = max(tprO - fprO);
    thrOOF = thrO(idxThrO);

    yPredOOF = double(scoresOOF >= thrOOF);

    % AUC by Sex (OOF)
    if ~isempty(sexAll)
        uniqSex = unique(sexAll);
        parts = strings(numel(uniqSex),1);
        for si = 1:numel(uniqSex)
            sx = uniqSex(si);
            mask = (sexAll == sx);
            if numel(unique(yVec(mask))) < 2
                aucSx = NaN;
            else
                [~,~,~,aucSx] = perfcurve(yVec(mask), scoresOOF(mask), 1);
            end
            parts(si) = sprintf('%s=%.3f', sx, aucSx);
        end
        aucBySexOOFStr = strjoin(parts, '; ');
    end

    % Subtype audit across ALL pathological samples
    if ~isempty(pathCodeAll)
        patMask = (yVec == 1) & (pathCodeAll >= 0);
        fnMask  = patMask & (yPredOOF == 0);

        codes = unique(pathCodeAll(patMask));
        nRows = numel(codes);

        codeOut = zeros(nRows,1,'int32');
        nameOut = strings(nRows,1);
        nPatOut = zeros(nRows,1);
        nFnOut  = zeros(nRows,1);
        fnRateOut = zeros(nRows,1);

        for ci = 1:nRows
            c = int32(codes(ci));
            msk = patMask & (pathCodeAll == c);
            nPat = sum(msk);
            nFn  = sum(fnMask & (pathCodeAll == c));

            codeOut(ci) = c;
            nameOut(ci) = pathology_name_from_map(pathologyMap, c);
            nPatOut(ci) = nPat;
            nFnOut(ci)  = nFn;
            fnRateOut(ci) = nFn / (nPat + eps);
        end

        auditOOF = table(codeOut, nameOut, nPatOut, nFnOut, fnRateOut, ...
            'VariableNames', {'Pathology_code','Full_pathology_name','N_pathological_total','N_false_negative','FN_rate'});
        auditOOF = sortrows(auditOOF, {'N_false_negative','FN_rate','N_pathological_total'}, {'descend','descend','descend'});

        auditDirOOF = fullfile(dataDir, 'subtype_error_audit_oof');
        if ~exist(auditDirOOF, 'dir'); mkdir(auditDirOOF); end
        subtypeAuditPath_OOF = fullfile(auditDirOOF, sprintf('SubtypeAudit_OOF_%s_%s_%s.csv', fsLabelToSuffix(fsLabel), groupName, bestKey));
        writetable(auditOOF, subtypeAuditPath_OOF);
    end
catch ME
    warning('OOF subtype audit failed: %s', ME.message);
end

% -----------------------------
% Subtype error audit (TEST pathological only)
% -----------------------------
subtypeAuditPath = "";
if ~isempty(pathCodeTest)
    patMask = (y_test == 1) & (pathCodeTest >= 0);
    fnMask = patMask & (yPred == 0);

    codes = unique(pathCodeTest(patMask));
    nRows = numel(codes);
    codeOut = zeros(nRows,1,'int32');
    nameOut = strings(nRows,1);
    nPatOut = zeros(nRows,1);
    nFnOut  = zeros(nRows,1);
    fnRateOut = zeros(nRows,1);

    for ci = 1:nRows
        c = int32(codes(ci));
        msk = patMask & (pathCodeTest == c);
        nPat = sum(msk);
        nFn  = sum(fnMask & (pathCodeTest == c));
        codeOut(ci) = c;
        nameOut(ci) = pathology_name_from_map(pathologyMap, c);
        nPatOut(ci) = nPat;
        nFnOut(ci)  = nFn;
        fnRateOut(ci) = nFn / (nPat + eps);
    end

    auditT = table(codeOut, nameOut, nPatOut, nFnOut, fnRateOut, ...
        'VariableNames', {'Pathology_code','Full_pathology_name','N_pathological_test','N_false_negative','FN_rate'});
    auditT = sortrows(auditT, {'FN_rate','N_pathological_test'}, {'descend','descend'});

    auditDir = fullfile(dataDir, 'subtype_error_audit');
    if ~exist(auditDir, 'dir'); mkdir(auditDir); end
    subtypeAuditPath = fullfile(auditDir, sprintf('SubtypeAudit_%s_%s_%s.csv', fsLabelToSuffix(fsLabel), groupName, bestKey));
    writetable(auditT, subtypeAuditPath);
end

% Save a small reviewer summary for this group
reviewerDir = fullfile(dataDir, 'reviewer_analysis');
if ~exist(reviewerDir, 'dir'); mkdir(reviewerDir); end
reviewerPath = fullfile(reviewerDir, sprintf('ReviewerAnalysis_%s_%s.csv', fsLabelToSuffix(fsLabel), groupName));
reviewerT = table(string(fsLabel), string(groupName), string(bestKey), bestAUC, thrBest, sens, spec, bacc, acc, string(cmPath), string(subtypeAuditPath), string(subtypeAuditPath_OOF), string(aucBySexStr), string(aucBySexOOFStr), ...
    'VariableNames', {'SamplingRate','Group','BestModelKey','TestAUC','Threshold_YoudenTrain','Sensitivity','Specificity','BalancedAcc','Accuracy','ConfusionMatrixPath','SubtypeAuditPath_Test','SubtypeAuditPath_OOF','AUC_by_Sex_Test','AUC_by_Sex_OOF'});
writetable(reviewerT, reviewerPath);
fprintf('  Reviewer analysis saved to: %s\n', reviewerPath);
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
% 4 subplots (one per MODEL). Each subplot overlays ROC curves for each
% FEATURE GROUP (noise/perturbation/tremor/complexity/all).

modelOrderPlot = {'logreg','svm_rbf','rf','mlp'};

prettyNameModel = struct();
prettyNameModel.logreg  = 'Logistic';
prettyNameModel.svm_rbf = 'SVM RBF';
prettyNameModel.rf      = 'Random Forest';
prettyNameModel.mlp     = 'MLP';

prettyNameGroup = struct();
prettyNameGroup.noise        = 'Noise';
prettyNameGroup.perturbation = 'Perturbation';
prettyNameGroup.tremor       = 'Tremor';
prettyNameGroup.complexity   = 'Complexity';
prettyNameGroup.all          = 'All';

f = figure('Position',[100 100 1000 1000]);
set(f,'PaperPositionMode','auto');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for mi = 1:numel(modelOrderPlot)
    mname = modelOrderPlot{mi};
    nexttile;
    hold on;

    hasAny = false;

    for gi = 1:numel(groupOrderForPlots)
        gname = groupOrderForPlots{gi};
        if isfield(rocStruct, gname) && isfield(rocStruct.(gname), mname)
            fpr = rocStruct.(gname).(mname).fpr;
            tpr = rocStruct.(gname).(mname).tpr;
            auc = rocStruct.(gname).(mname).testAUC;

            plot(fpr, tpr, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('%s (AUC=%.2f)', prettyNameGroup.(gname), auc));
            hasAny = true;
        end
    end

    plot([0 1],[0 1],'k--','LineWidth',1);
    axis square;
    xlim([0 1]); ylim([0 1]);
    xlabel('False positive rate');
    ylabel('True positive rate');
    title(sprintf('%s – %s', prettyNameModel.(mname), fsLabel));

    if hasAny
        legend('Location','SouthEast','Box','off');
    else
        text(0.5, 0.5, 'No ROC data', 'HorizontalAlignment', 'center');
    end

    hold off;
end


    % ==== SAVE FIG ====
    if contains(csvPath, '50kHz')
        fs_suffix = '50kHz';
    elseif contains(csvPath, '25kHz')
        fs_suffix = '25kHz';
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
    %   ROC_HUPA_25kHz_MATLAB
    
    png_path = fullfile(fig_dir, [file_base '.png']);
    pdf_path = fullfile(fig_dir, [file_base '.pdf']);
    
    saveas(f, png_path);
    saveas(f, pdf_path);
    
    fprintf('\nROC figure saved to:\n  %s\n  %s\n', png_path, pdf_path);

end % end loop over datasets

%% ===================== 8) MODEL SUBFUNCTIONS ============================

function [cvAUC, testAUC, fprTest, tprTest, scoresTest, thrYouden] = model_logreg(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Logistic Regression using 'fitclinear'.
% Tunes: Lambda (Regularization strength) and Regularization type (Ridge/Lasso).
% Returns:
%   - cvAUC: mean AUC across inner folds
%   - testAUC: AUC on hold-out test
%   - scoresTest: posterior score/probability for class 1 on test
%   - thrYouden: threshold selected on TRAIN using Youden's J (train scores)

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
            scores = scoreVal(:,2);
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

% --- Threshold on TRAIN (Youden's J) ---
[~, scoreTrain] = predict(mdlBest, XtrAll);
scoresTrain = scoreTrain(:,2);
[fprTr, tprTr, thrTr] = perfcurve(ytrain, scoresTrain, 1);
[~, idxThr] = max(tprTr - fprTr);
thrYouden = thrTr(idxThr);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(mdlBest, XtestZ);
scoresTest = scoreTest(:,2);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);

cvAUC = bestCvAUC;
end

function [cvAUC, testAUC, fprTest, tprTest, scoresTest, thrYouden] = model_svm_rbf(Xtrain, ytrain, Xtest, ytest)
% Grid Search for SVM with RBF Kernel.
% Tunes: BoxConstraint (C) and KernelScale.
% Uses 'fitPosterior' to get probability estimates.
% Returns:
%   - scoresTest: posterior probability for class 1 on test
%   - thrYouden: threshold selected on TRAIN using Youden's J (train scores)

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

mdlBest = fitPosterior(mdlBest, XtrAll, ytrain);

% --- Threshold on TRAIN (Youden's J) ---
[~, scoreTrain] = predict(mdlBest, XtrAll);
scoresTrain = scoreTrain(:,2);
[fprTr, tprTr, thrTr] = perfcurve(ytrain, scoresTrain, 1);
[~, idxThr] = max(tprTr - fprTr);
thrYouden = thrTr(idxThr);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(mdlBest, XtestZ);
scoresTest = scoreTest(:,2);
[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);

cvAUC = bestCvAUC;
end

function [cvAUC, testAUC, fprTest, tprTest, scoresTest, thrYouden] = model_random_forest(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Random Forest using 'TreeBagger'.
% Tunes: Number of Trees and MinLeafSize.
% Returns:
%   - scoresTest: estimated probability for class 1 on test
%   - thrYouden: threshold selected on TRAIN using Youden's J (train scores)

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

            classNames = mdl.ClassNames;
            if isa(classNames,'categorical')
                posIdx = find(classNames == categorical(1));
            else
                posIdx = find(strcmp(classNames, '1'));
            end
            if isempty(posIdx); posIdx = 2; end

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

% --- Threshold on TRAIN (Youden's J) ---
[~, scoreTrain] = predict(mdlBest, Xtrain);
classNamesTr = mdlBest.ClassNames;
if isa(classNamesTr,'categorical')
    posIdxTr = find(classNamesTr == categorical(1));
else
    posIdxTr = find(strcmp(classNamesTr, '1'));
end
if isempty(posIdxTr); posIdxTr = 2; end
scoresTrain = scoreTrain(:,posIdxTr);
[fprTr, tprTr, thrTr] = perfcurve(ytrain, scoresTrain, 1);
[~, idxThr] = max(tprTr - fprTr);
thrYouden = thrTr(idxThr);

% --- Evaluate on Test Set ---
[~, scoreTest] = predict(mdlBest, Xtest);
classNames = mdlBest.ClassNames;
if isa(classNames,'categorical')
    posIdx = find(classNames == categorical(1));
else
    posIdx = find(strcmp(classNames, '1'));
end
if isempty(posIdx); posIdx = 2; end
scoresTest = scoreTest(:,posIdx);

[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);

cvAUC = bestCvAUC;
end

function [cvAUC, testAUC, fprTest, tprTest, scoresTest, thrYouden] = model_mlp(Xtrain, ytrain, Xtest, ytest)
% Grid Search for Shallow Neural Network (MLP) using 'fitcnet'.
% Tunes: Layer Sizes and Lambda (Regularization).
% Returns:
%   - scoresTest: posterior probability for class 1 on test
%   - thrYouden: threshold selected on TRAIN using Youden's J (train scores)

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
            if isempty(posIdx); posIdx = 2; end

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

% --- Threshold on TRAIN (Youden's J) ---
[~, scoreTrain] = predict(MdlBest, XtrAll);
classNamesTr = MdlBest.ClassNames;
if isnumeric(classNamesTr)
    posIdxTr = find(classNamesTr == 1);
else
    posIdxTr = find(strcmp(string(classNamesTr), '1'));
end
if isempty(posIdxTr); posIdxTr = 2; end
scoresTrain = scoreTrain(:,posIdxTr);
[fprTr, tprTr, thrTr] = perfcurve(ytrain, scoresTrain, 1);
[~, idxThr] = max(tprTr - fprTr);
thrYouden = thrTr(idxThr);

% --- Evaluate on Test Set ---
XtestZ = (Xtest - muAll) ./ sigmaAll;
[~, scoreTest] = predict(MdlBest, XtestZ);
classNames = MdlBest.ClassNames;
if isnumeric(classNames)
    posIdx = find(classNames == 1);
else
    posIdx = find(strcmp(string(classNames), '1'));
end
if isempty(posIdx); posIdx = 2; end
scoresTest = scoreTest(:,posIdx);

[fprTest, tprTest, ~, testAUC] = perfcurve(ytest, scoresTest, 1);

cvAUC = bestCvAUC;
end

function m = load_pathology_map(xlsxPath)
    % Returns containers.Map(int32 -> char) mapping pathology code -> FULL pathology name.
    % Full name is built from hierarchical dotted codes by concatenating prefix names with ' > '.

    Tm = readtable(xlsxPath, 'Sheet', 'Pathology classification');
    if ~ismember('Code', Tm.Properties.VariableNames) || ~ismember('Pathology', Tm.Properties.VariableNames)
        error('Expected columns "Code" and "Pathology" in sheet "Pathology classification".');
    end

    codeRaw = string(Tm.Code);
    nameRaw = string(Tm.Pathology);

    % dotted -> name (includes internal nodes)
    dottedMap = containers.Map('KeyType','char','ValueType','char');
    for i = 1:numel(codeRaw)
        c = strtrim(codeRaw(i));
        if c == "" || lower(c) == "nan"
            continue;
        end
        c2 = regexprep(c, '[^0-9\.]', '');
        c2 = strip(c2, '.');
        if c2 == ""
            continue;
        end
        dottedMap(char(c2)) = char(strtrim(nameRaw(i)));
    end

    m = containers.Map('KeyType','int32','ValueType','char');
    for i = 1:numel(codeRaw)
        c = strtrim(codeRaw(i));
        if c == "" || lower(c) == "nan"
            continue;
        end
        c2 = regexprep(c, '[^0-9\.]', '');
        c2 = strip(c2, '.');
        if c2 == ""
            continue;
        end

        codeDigits = regexprep(c2, '\D', '');
        codeNum = str2double(codeDigits);
        if isnan(codeNum)
            continue;
        end
        k = int32(codeNum);

        toks = split(c2, '.');
        parts = strings(0,1);
        for j = 1:numel(toks)
            pref = strjoin(toks(1:j), '.');
            if isKey(dottedMap, char(pref))
                nm = string(dottedMap(char(pref)));
                if nm ~= "" && (isempty(parts) || parts(end) ~= nm)
                    parts(end+1,1) = nm; %#ok<AGROW>
                end
            end
        end

        fullName = "NR";
        if ~isempty(parts)
            fullName = strjoin(parts, ' > ');
        end

        if ~isKey(m, k)
            m(k) = char(fullName);
        end
    end
end

function name = pathology_name_from_map(m, code)
    if isempty(m) || m.Count == 0
        name = "NR";
        return;
    end
    if isKey(m, int32(code))
        name = string(m(int32(code)));
    else
        name = "NR";
    end
end

function [sens, spec, bacc, acc] = confusion_metrics_from_cm(cm)
    tn = cm(1,1);
    fp = cm(1,2);
    fn = cm(2,1);
    tp = cm(2,2);

    sens = tp / (tp + fn + eps);
    spec = tn / (tn + fp + eps);
    bacc = 0.5 * (sens + spec);
    acc  = (tp + tn) / (tp + tn + fp + fn + eps);
end

function save_confusion_matrix_figure(cm, titleStr, outPath)
    f = figure('Visible','off');
    imagesc(cm);
    axis square;
    xlabel('Predicted');
    ylabel('True');
    title(titleStr);

    xticks([1 2]); yticks([1 2]);
    xticklabels({'Healthy','Pathological'});
    yticklabels({'Healthy','Pathological'});

    for i = 1:2
        for j = 1:2
            text(j, i, num2str(cm(i,j)), 'HorizontalAlignment','center', 'Color','w', 'FontWeight','bold');
        end
    end

    set(gca,'TickLength',[0 0]);
    if exist('exportgraphics','file') == 2
        exportgraphics(f, outPath, 'Resolution', 300);
    else
        % Fallback for older MATLAB versions
        print(f, outPath, '-dpng', '-r300');
    end
    close(f);
end

function suf = fsLabelToSuffix(fsLabel)
    if contains(fsLabel, '50')
        suf = '50kHz';
    elseif contains(fsLabel, '25')
        suf = '25kHz';
    else
        suf = 'unknownFS';
    end
end

