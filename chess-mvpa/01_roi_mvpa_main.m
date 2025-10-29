%% ========================================================================
%% ROI-based Decoding and RSA (CoSMoMVPA)
%%
%% Purpose
%% - Per-subject ROI decoding (SVM) and ROI RSA correlations for the 3 main
%%   dimensions used throughout the repository:
%%     1) checkmate (binary: checkmate vs non-checkmate)
%%     2) visualStimuli (20-class: merged visual identity ignoring '(nomate)')
%%     3) categories (multi-class across all 40 boards; keep original labels)
%%
%% Output
%% - Subject-level TSV files saved under:
%%     <BIDS_DERIVATIVES>/mvpa/<timestamp>_glasser_regions_bilateral/
%%       ├── svm/sub-XX/mvpa_cv.tsv          (decoding accuracy per ROI per target)
%%       └── rsa_corr/sub-XX/rsa_corr.tsv    (RSA r-values per ROI per model)
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
% BIDS derivatives root (override with env CHESS_BIDS_DERIVATIVES)
defaultDeriv = '/data/projects/chess/data/BIDS/derivatives';
derivativesDir = getenv_default('CHESS_BIDS_DERIVATIVES', defaultDeriv);

% GLM root for unsmoothed SPM outputs (subject folders live here)
% Matches old-implementation logic (unsmoothed GLM for MVPA)
glmRoot = fullfile(derivativesDir, 'fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked', 'MNI', 'fmriprep-SPM-MNI', 'GLM');

% ROI atlas and metadata (override with CHESS_ROI_ATLAS_22 and CHESS_ROI_TSV_22)
roiAtlas = getenv_default('CHESS_ROI_ATLAS_22', ...
    fullfile(derivativesDir, 'rois', 'glasser22', 'tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii'));
roiTSV = getenv_default('CHESS_ROI_TSV_22', ...
    fullfile(derivativesDir, 'rois', 'glasser22', 'region_info.tsv'));

% Output root
ts = datestr(now, 'yyyymmdd-HHMMSS');
outRoot = fullfile(derivativesDir, 'mvpa', [ts, '_glasser_regions_bilateral']);
outRootSVM = fullfile(outRoot, 'svm');
outRootRSACorr = fullfile(outRoot, 'rsa_corr');
mkdir_p(outRootSVM); mkdir_p(outRootRSACorr);

fprintf('[INFO] Outputs will be written under: %s\n', outRoot);

%% --------------------------- Subject discovery ---------------------------
subDirs = find_subjects(glmRoot, 'sub-*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmRoot);

%% --------------------------- Load ROIs -----------------------------------
[Vroi, roi_data] = spm_read_vols(spm_vol(roiAtlas)); %#ok<ASGLU>
roiInfo = readtable(roiTSV, 'FileType','text', 'Delimiter','\t');
if ~ismember('index', roiInfo.Properties.VariableNames)
    error('ROI TSV must contain an "index" column with integer labels.');
end

region_ids = roiInfo.index(:)';
region_names = roiInfo.name(:)';
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
    [checkmateVec, categoriesVec, stimVec, visStimVec] = parse_label_regressors(ds);

    % Define regressor catalog reflecting original logic
    % targets: vector across all samples (all runs)
    regressors = struct();
    regressors.checkmate.targets = checkmateVec;   % 2-class
    regressors.categories.targets = categoriesVec; % multi-class across 40 boards
    regressors.visualStimuli.targets = visStimVec; % 20-class

    % Stimulus-based targets to average for RSA (full set)
    regressors.stimuli.targets = stimVec;

    % RSA model RDMs computed on first run (chunk==1) subset
    firstRunMask = (ds.sa.chunks == 1);
    regressors.checkmate.rdm = compute_rdm(checkmateVec(firstRunMask), 'similarity');
    regressors.categories.rdm = compute_rdm(categoriesVec(firstRunMask), 'similarity');
    regressors.visualStimuli.rdm = compute_rdm(visStimVec(firstRunMask), 'similarity');

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

    % ---------------------- ROI loop ----------------------
    for r = 1:nROI
        rid = region_ids(r);
        mask = (roi_data == rid);
        if ~any(mask(:))
            continue; % empty ROI
        end

        ds_slice = cosmo_slice(ds, mask, 2);
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
    svm_tbl = array2table(svm_mat, 'VariableNames', matlab_safe_names(region_names));
    svm_tbl = addvars(svm_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    writetable(svm_tbl, fullfile(subOutSVM, 'mvpa_cv.tsv'), 'FileType','text', 'Delimiter','\t');

    rsa_tbl = array2table(rsa_mat, 'VariableNames', matlab_safe_names(region_names));
    rsa_tbl = addvars(rsa_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    writetable(rsa_tbl, fullfile(subOutRSA, 'rsa_corr.tsv'), 'FileType','text', 'Delimiter','\t');

    fprintf('[INFO]   Saved SVM and RSA TSV files for %s\n', subName);
end

fprintf('\n[INFO] Done. Subject-level TSV files written to: %s\n', outRoot);

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

function [checkmateVec, categoriesVec, stimVec, visStimVec] = parse_label_regressors(ds)
    labels = ds.sa.labels(:);

    % Checkmate C/NC → 2 levels encoded as 2/1
    cmLabels = regexp(labels, '(?<=\s)(C|NC)\d+', 'match', 'once');
    checkmateVec = cellfun(@(x) strcmpi(x(1), 'C') + 1, cmLabels);

    % Categories: concatenated (C|NC)(\d+) mapped to dense integers (across all 40)
    catTokens = regexp(labels, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once');
    concatCats = cellfun(@(x) [x{1}, x{2}], catTokens, 'UniformOutput', false);
    uniqCats = unique(concatCats, 'stable');
    catMap = containers.Map(uniqCats, 1:numel(uniqCats));
    categoriesVec = cellfun(@(x) catMap(x), concatCats);

    % Stimulus strings between '_' and '*', lowercased
    stimLabels = regexp(labels, '(?<=_).*?(?=\*)', 'match', 'once');
    lowerStim = lower(stimLabels);
    uStim = unique(lowerStim, 'stable');
    stimMap = containers.Map(uStim, 1:numel(uStim));
    stimVec = cellfun(@(x) stimMap(x), lowerStim);

    % Visual identity ignoring '(nomate)' → 20 classes
    cleanStim = erase(lowerStim, '(nomate)');
    uVis = unique(cleanStim, 'stable');
    visMap = containers.Map(uVis, 1:numel(uVis));
    visStimVec = cellfun(@(x) visMap(x), cleanStim);
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
