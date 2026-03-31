%% ========================================================================
%% ROI-based Decoding and RSA -- RUN-MATCHED CONTROL
%%
%% PURPOSE
%% Reviewer R2 (Major Comment 3) noted that experts completed fewer runs
%% on average (M=8.4, SD=2.0) than novices (M=9.5, SD=1.4; p=0.06) and
%% requested a control analysis equating run counts across groups.
%%
%% APPROACH
%% We match the novice run distribution to the expert distribution so both
%% groups have identical run counts: 8 subjects x 6 runs + 12 subjects x
%% 10 runs = 168 total runs per group. Expert data is fully preserved.
%% For novices, the 8 subjects with the fewest runs (or lowest IDs among
%% those with 10 runs) are capped at 6 runs. The remaining 12 novices
%% keep all 10 runs. This is deterministic (no random selection).
%%
%% Matched novices (capped at 6 runs):
%%   sub-01 (had 6), sub-02 (had 6), sub-39 (had 7),
%%   sub-15, sub-17, sub-18, sub-19, sub-21 (had 10 each)
%%
%% MODIFICATIONS vs chess-mvpa/01_roi_mvpa_main.m
%%   1. Run-capping filter applied after dataset loading (~line 93)
%%   2. Output dirs changed to mvpa-rsa-run8/ and mvpa-decoding-run8/
%%
%% Everything else is identical: same ROI atlas, RSA measure, SVM
%% classifier, targets, and helper functions.
%%
%% Dependencies
%% - CoSMoMVPA on MATLAB path
%% - SPM on MATLAB path (for cosmo_fmri_dataset and cosmo_map2fmri)
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------
% Central config (edit common/chess_config.m to change paths)
addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'common'));
cfg = chess_config();

derivativesDir = cfg.derivatives;
glmRoot        = cfg.glmUnsmoothed;
roiAtlas       = cfg.roiGlasser22Atlas;
roiTSV         = cfg.roiGlasser22TSV;

% Output root: save results in the repo, not in BIDS derivatives.
scriptDir = fileparts(mfilename('fullpath'));
outRootRSACorr = fullfile(scriptDir, 'results', 'mvpa-rsa-run-matched');
mkdir_p(outRootRSACorr);

fprintf('[INFO] RSA outputs will be written under: %s\n', outRootRSACorr);

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

    % --- Run-matching control (ONLY CHANGE vs original script) ---
    %
    % Expert run distribution: 8 subjects have 6 runs, 12 have 10 runs.
    % Novice run distribution: 2 subjects have 6 runs, 1 has 7, 17 have 10.
    %
    % To equate: we cap 8 novices at 6 runs (matching the 8 six-run experts).
    % The 2 novices who already have 6 runs are included naturally.
    % sub-39 (7 runs) is capped to 6. Five ten-run novices (lowest IDs:
    % sub-15, sub-17, sub-18, sub-19, sub-21) are capped to 6.
    % The remaining 12 ten-run novices keep all 10 runs.
    %
    % Result: both groups have identical distributions (8x6 + 12x10 = 168).
    % Expert data is completely untouched.
    %
    % Filtering uses ds.sa.chunks (run indices, 1-indexed). Keeping
    % chunks <= 6 retains the first 6 runs.
    novices_cap6 = {'sub-01','sub-02','sub-15','sub-17','sub-18','sub-19','sub-21','sub-39'};
    n_runs_orig = numel(unique(ds.sa.chunks));
    if ismember(subName, novices_cap6)
        max_runs = 6;
        run_mask = ds.sa.chunks <= max_runs;
        ds = cosmo_slice(ds, run_mask, 1);
        ds = cosmo_remove_useless_data(ds);
    end
    fprintf('  Run-matched: using %d of %d runs\n', ...
        numel(unique(ds.sa.chunks)), n_runs_orig);

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

    % Prepare output directory (RSA only -- no decoding in this control)
    subOutRSA = fullfile(outRootRSACorr, subName); mkdir_p(subOutRSA);

    % Initialize result table (rows=targets, cols=rois)
    targetNames = fieldnames(regressors);
    targetNames = setdiff(targetNames, {'stimuli'}); % exclude helper
    nReg = numel(targetNames);
    nROI = numel(region_ids);

    rsa_mat = nan(nReg, nROI);

    % RSA measure (same as original: cosmo_target_dsm_corr_measure with centering)
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

        % Average patterns across runs by stimulus identity (same as original)
        ds_rsa = ds_slice;
        ds_rsa.sa.targets = regressors.stimuli.targets;
        ds_rsa_averaged = cosmo_fx(ds_rsa, @(x) mean(x,1), 'targets');

        % Correlate neural RDM with each model RDM
        for t = 1:nReg
            tname = targetNames{t};
            try
                rsa_args.target_dsm = regressors.(tname).rdm;
                res = rsa_measure(ds_rsa_averaged, rsa_args);
                if ~isempty(res.samples)
                    rsa_mat(t, r) = res.samples; % scalar Pearson r
                end
            catch ME
                fprintf('[WARN]   RSA failed for %s, ROI %d: %s\n', tname, rid, ME.message);
            end
        end
    end

    % ---------------------- Save RSA TSV output ----------------------
    rsa_tbl = array2table(rsa_mat, 'VariableNames', matlab_safe_names(region_names));
    rsa_tbl = addvars(rsa_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    rsaFilename = sprintf('%s_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv', subName);
    writetable(rsa_tbl, fullfile(subOutRSA, rsaFilename), 'FileType','text', 'Delimiter','\t');

    fprintf('[INFO]   Saved RSA TSV for %s\n', subName);
end

fprintf('\n[INFO] Done. Subject-level RSA files written to:\n');
fprintf('         RSA: %s\n', outRootRSACorr);

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

    % Checkmate C/NC -> 2 levels encoded as 2/1
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

    % Visual similarity: identity ignoring '(nomate)' -> 20 classes
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
