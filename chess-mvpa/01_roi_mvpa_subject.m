%% ========================================================================
%% ROI-based Decoding and RSA (CoSMoMVPA)
%%
%% Purpose
%% - Per-subject ROI decoding (SVM) and ROI RSA correlations for the main
%%   dimensions (using all 40 boards):
%%     1) checkmate (binary: checkmate vs non-checkmate)
%%     2) visual_similarity (20-class: merged visual identity ignoring '(nomate)')
%%     3) strategy (multi-class across all 40 boards; keep original labels)
%%
%% Output
%% - Subject-level TSV files saved under:
%%     <BIDS_DERIVATIVES>/
%%       ├── mvpa-decoding/sub-XX/sub-XX_space-MNI152NLin2009cAsym_roi-glasser_accuracy.tsv
%%       └── mvpa-rsa/sub-XX/sub-XX_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv
%% - Each file contains multiple target rows (one row per target dimension)
%% - Fine-grained targets (from supplementary/mvpa-finer) appear as additional rows
%%   with "_half" suffix (using only 20 checkmate boards)
%%
%% Notes
%% - This script preserves the original analysis logic from old-implementation
%%   (mvpa_rsa_all_regressors.m) with a cleaner structure and outputs.
%% - No plotting here — analysis scripts only produce artifacts. Downstream
%%   Python scripts will aggregate at group level and plot/report.
%% - Paths may be overridden using environment variables for portability.
%%
%% Dependencies
%% - CoSMoMVPA on MATLAB path
%% - SPM on MATLAB path (for cosmo_fmri_dataset and cosmo_map2fmri)
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------
% Central config (edit common/chess_config.m to change paths)
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'common'));
cfg = chess_config();

derivativesDir = cfg.derivatives;
glmRoot        = cfg.spmUnsmoothed;
roiAtlas       = cfg.roiGlasser22Atlas;
roiTSV         = cfg.roiGlasser22TSV;
outRootSVM     = cfg.mvpaDecoding;
outRootRSACorr = cfg.mvpaRsa;
mkdir_p(outRootSVM); mkdir_p(outRootRSACorr);

fprintf('[INFO] Outputs will be written under: %s and %s\n', outRootSVM, outRootRSACorr);

%% --------------------------- Subject discovery ---------------------------
subDirs = find_subjects(glmRoot, 'sub-*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmRoot);

%% --------------------------- Load ROIs -----------------------------------
roi_data = spm_read_vols(spm_vol(roiAtlas));
roiInfo = readtable(roiTSV, 'FileType','text', 'Delimiter','\t');
% Support both column naming conventions (index/name or ROI_idx/roi_name)
if ismember('index', roiInfo.Properties.VariableNames)
    region_ids = roiInfo.index(:)';
    region_names = roiInfo.name(:)';
elseif ismember('ROI_idx', roiInfo.Properties.VariableNames)
    region_ids = roiInfo.ROI_idx(:)';
    region_names = roiInfo.roi_name(:)';
else
    error('ROI TSV must contain an "index" or "ROI_idx" column with integer labels.');
end
fprintf('[INFO] Loaded ROI atlas with %d labeled regions\n', numel(region_ids));

%% ========================= Main subject loop =============================
for s = 1:numel(subDirs)
    subName = subDirs(s).name;  % e.g., 'sub-01'
    fprintf('\n[INFO] Processing %s\n', subName);

    spmSubjDir = fullfile(glmRoot, subName, 'exp');
    spmMat = fullfile(spmSubjDir, 'SPM.mat');
    if ~exist(spmMat, 'file')
        fprintf('[WARN]   Missing SPM.mat for %s, skipping.\n', subName);
        continue;
    end

    % Load dataset from SPM model (beta estimates)
    ds = cosmo_fmri_dataset(spmMat);
    ds = cosmo_remove_useless_data(ds);
    cosmo_check_dataset(ds);
    if isempty(ds.samples)
        fprintf('[WARN]   Empty dataset for %s, skipping.\n', subName);
        continue;
    end

    % --------------------------------------------------------------------
    % Parse label strings to regressors and model vectors
    % --------------------------------------------------------------------
    [checkmateVec, strategyVec, stimVec, visSimilarityVec] = parse_label_regressors(ds);

    % Define regressor catalog reflecting original logic
    % targets: vector across all samples (all runs)
    regressors = struct();
    regressors.checkmate.targets = checkmateVec;   % 2-class
    regressors.strategy.targets = strategyVec; % multi-class across 40 boards
    regressors.visual_similarity.targets = visSimilarityVec; % 20-class

    % Stimulus-based targets to average for RSA (full set)
    regressors.stimuli.targets = stimVec;

    % RSA model RDMs computed on first run (chunk==1) subset
    firstRunMask = (ds.sa.chunks == 1);
    regressors.checkmate.rdm = compute_rdm(checkmateVec(firstRunMask), 'similarity');
    regressors.strategy.rdm = compute_rdm(strategyVec(firstRunMask), 'similarity');
    regressors.visual_similarity.rdm = compute_rdm(visSimilarityVec(firstRunMask), 'similarity');

    % Prepare outputs
    subOutSVM = fullfile(outRootSVM, subName); mkdir_p(subOutSVM);
    subOutRSA = fullfile(outRootRSACorr, subName); mkdir_p(subOutRSA);

    % Initialize result tables (rows=targets, cols=rois)
    targetNames = fieldnames(regressors);
    targetNames = setdiff(targetNames, {'stimuli'}); % exclude helper
    nReg = numel(targetNames);
    nROI = numel(region_ids);

    svm_mat = nan(nReg, nROI);
    rsa_mat = nan(nReg, nROI);

    % Measures
    classifier = @cosmo_classify_svm;
    rsa_measure = @cosmo_target_dsm_corr_measure;
    rsa_args = struct('center_data', true);

    % Precompute feature-space ROI membership from 3D atlas.
    % ds.fa.i/j/k are 1-indexed voxel coordinates for each feature in ds.
    % We look up each feature's ROI label from the 3D atlas volume.
    ijk = [ds.fa.i; ds.fa.j; ds.fa.k];  % 3 x n_features
    n_feat = size(ijk, 2);
    feat_roi = zeros(1, n_feat);
    for fi = 1:n_feat
        feat_roi(fi) = roi_data(ijk(1,fi), ijk(2,fi), ijk(3,fi));
    end

    % ---------------------- ROI loop ----------------------
    for r = 1:nROI
        rid = region_ids(r);
        feat_mask = (feat_roi == rid);
        if sum(feat_mask) == 0
            continue; % empty ROI
        end

        ds_slice = cosmo_slice(ds, feat_mask, 2);
        ds_slice = cosmo_remove_useless_data(ds_slice);
        if isempty(ds_slice.samples) || size(ds_slice.samples,2) < 6
            continue;
        end

        % For RSA, build averaged slice by stimulus identity
        ds_rsa = ds_slice;
        ds_rsa.sa.targets = regressors.stimuli.targets;
        ds_rsa_averaged = cosmo_fx(ds_rsa, @(x) mean(x,1), 'targets');

        % Iterate over decoding/RSA targets
        for t = 1:nReg
            tname = targetNames{t};

            % ---------- SVM decoding ----------
            ds_dec = ds_slice;
            ds_dec.sa.targets = regressors.(tname).targets;

            if numel(unique(ds_dec.sa.targets)) >= 2
                parts = cosmo_nfold_partitioner(ds_dec);
                parts = cosmo_balance_partitions(parts, ds_dec, 'nmin', 1);
                try
                    [~, acc] = cosmo_crossvalidate(ds_dec, classifier, parts);
                    svm_mat(t, r) = acc;
                catch ME
                    fprintf('[WARN]   SVM failed for %s, ROI %d: %s\n', tname, rid, ME.message);
                end
            end

            % ---------- RSA correlation ----------
            try
                rsa_args.target_dsm = regressors.(tname).rdm; % N x N
                res = rsa_measure(ds_rsa_averaged, rsa_args);
                if ~isempty(res.samples)
                    rsa_mat(t, r) = res.samples; % scalar r
                end
            catch ME
                fprintf('[WARN]   RSA failed for %s, ROI %d: %s\n', tname, rid, ME.message);
            end
        end
    end

    % ---------------------- Save TSV outputs ----------------------
    % BIDS naming with explicit stat- entity:
    %   sub-XX_space-MNI152NLin2009cAsym_roi-glasser_stat-accuracy_decoding.tsv
    %   sub-XX_space-MNI152NLin2009cAsym_roi-glasser_stat-r_rsa.tsv
    svm_tbl = array2table(svm_mat, 'VariableNames', matlab_safe_names(region_names));
    svm_tbl = addvars(svm_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    svmFilename = sprintf('%s_space-MNI152NLin2009cAsym_roi-glasser_stat-accuracy_decoding.tsv', subName);
    writetable(svm_tbl, fullfile(subOutSVM, svmFilename), 'FileType','text', 'Delimiter','\t');

    rsa_tbl = array2table(rsa_mat, 'VariableNames', matlab_safe_names(region_names));
    rsa_tbl = addvars(rsa_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    rsaFilename = sprintf('%s_space-MNI152NLin2009cAsym_roi-glasser_stat-r_rsa.tsv', subName);
    writetable(rsa_tbl, fullfile(subOutRSA, rsaFilename), 'FileType','text', 'Delimiter','\t');

    fprintf('[INFO]   Saved SVM and RSA TSV files for %s\n', subName);
end

fprintf('\n[INFO] Done. Subject-level TSV files written to:\n');
fprintf('         Decoding: %s\n', outRootSVM);
fprintf('         RSA:      %s\n', outRootRSACorr);

%% ========================================================================
%% Helper functions (kept local for portability)
%% ========================================================================

function out = getenv_default(name, default)
    val = getenv(name);
    if isempty(val)
        out = default;
    else
        out = val;
    end
end

function mkdir_p(p)
    if ~exist(p, 'dir'), mkdir(p); end
end

function d = find_subjects(root, pattern)
    d = dir(fullfile(root, pattern));
    d = d([d.isdir]);
end

function names = matlab_safe_names(cellstr_in)
    names = matlab.lang.makeValidName(cellstr_in, 'ReplacementStyle','delete');
end

function [checkmateVec, strategyVec, stimVec, visSimilarityVec] = parse_label_regressors(ds)
    labels = ds.sa.labels(:);

    % Checkmate C/NC → 2 levels encoded as 2/1
    cmLabels = regexp(labels, '(?<=\s)(C|NC)\d+', 'match', 'once');
    checkmateVec = cellfun(@(x) strcmpi(x(1), 'C') + 1, cmLabels);

    % Strategy: concatenated (C|NC)(\d+) mapped to dense integers (across all 40)
    catTokens = regexp(labels, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once');
    concatCats = cellfun(@(x) [x{1}, x{2}], catTokens, 'UniformOutput', false);
    uniqCats = unique(concatCats, 'stable');
    catMap = containers.Map(uniqCats, 1:numel(uniqCats));
    strategyVec = cellfun(@(x) catMap(x), concatCats);

    % Stimulus strings between '_' and '*', lowercased
    stimLabels = regexp(labels, '(?<=_).*?(?=\*)', 'match', 'once');
    lowerStim = lower(stimLabels);
    uStim = unique(lowerStim, 'stable');
    stimMap = containers.Map(uStim, 1:numel(uStim));
    stimVec = cellfun(@(x) stimMap(x), lowerStim);

    % Visual similarity: identity ignoring '(nomate)' → 20 classes
    cleanStim = erase(lowerStim, '(nomate)');
    uVis = unique(cleanStim, 'stable');
    visMap = containers.Map(uVis, 1:numel(uVis));
    visSimilarityVec = cellfun(@(x) visMap(x), cleanStim);
end

function RDM = compute_rdm(vec, metric)
    v = vec(:);
    switch lower(metric)
        case 'similarity'
            RDM = double(bsxfun(@ne, v, v'));
        otherwise
            error('Unsupported RDM metric: %s', metric);
    end
end
