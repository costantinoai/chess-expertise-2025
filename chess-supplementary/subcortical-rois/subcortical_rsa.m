%% ========================================================================
%% Subcortical ROI-based Decoding and RSA (CoSMoMVPA) - CAB-NP Atlas
%%
%% Purpose
%% - Per-subject ROI decoding (SVM) and RSA correlations for subcortical
%%   ROIs using the Cole-Anticevic Brain-wide Network Partition (CAB-NP).
%% - Mirrors chess-mvpa/01_roi_mvpa_main.m exactly, with only:
%%     1) CAB-NP subcortical bilateral atlas instead of Glasser-22
%%     2) Outputs to mvpa-rsa-subcortical/ and mvpa-decoding-subcortical/
%%
%% Dimensions tested (all 40 boards):
%%   1) checkmate (binary: checkmate vs non-checkmate)
%%   2) visual_similarity (20-class: merged visual identity ignoring '(nomate)')
%%   3) strategy (multi-class across all 40 boards; keep original labels)
%%
%% Output
%% - Subject-level TSV files saved under:
%%     <BIDS_DERIVATIVES>/
%%       mvpa-decoding-subcortical/sub-XX/sub-XX_..._roi-cabnp_accuracy.tsv
%%       mvpa-rsa-subcortical/sub-XX/sub-XX_..._roi-cabnp_rdm.tsv
%% - Each file contains multiple target rows (one row per target dimension)
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

% ROI atlas: CAB-NP subcortical bilateral atlas (resampled to functional space)
% Use .nii (uncompressed) since SPM's spm_vol does not support .nii.gz
roiAtlas = cfg.roiCabnpAtlas;
roiTSV   = cfg.roiCabnpTSV;

% Output roots: subcortical SVM and RSA results
outRootSVM = fullfile(derivativesDir, 'mvpa-decoding-subcortical');
outRootRSACorr = fullfile(derivativesDir, 'mvpa-rsa-subcortical');
mkdir_p(outRootSVM); mkdir_p(outRootRSACorr);

fprintf('[INFO] ROI atlas: %s\n', roiAtlas);
fprintf('[INFO] ROI metadata: %s\n', roiTSV);
fprintf('[INFO] Outputs SVM: %s\n', outRootSVM);
fprintf('[INFO] Outputs RSA: %s\n\n', outRootRSACorr);

%% --------------------------- Subject discovery ---------------------------
subDirs = find_subjects(glmRoot, 'sub-*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmRoot);

%% --------------------------- Load ROIs -----------------------------------
roi_data = spm_read_vols(spm_vol(roiAtlas));
roiInfo = readtable(roiTSV, 'FileType','text', 'Delimiter','\t');

% Support both 'ROI_idx' and 'index' column names
if ismember('ROI_idx', roiInfo.Properties.VariableNames)
    region_ids = roiInfo.ROI_idx(:)';
elseif ismember('index', roiInfo.Properties.VariableNames)
    region_ids = roiInfo.index(:)';
else
    error('ROI TSV must contain an "ROI_idx" or "index" column with integer labels.');
end
region_names = roiInfo.roi_name(:)';
fprintf('[INFO] Loaded ROI atlas with %d labeled regions\n', numel(region_ids));
for r = 1:numel(region_ids)
    fprintf('  ROI %d: %s (%d voxels)\n', region_ids(r), region_names{r}, sum(roi_data(:) == region_ids(r)));
end

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

    % Decompress .nii.gz betas to a temporary directory for CoSMoMVPA
    % (CoSMoMVPA's NIfTI loader does not support .nii.gz)
    gzFiles = dir(fullfile(spmSubjDir, '*.nii.gz'));
    tmpSubjDir = '';
    if ~isempty(gzFiles)
        tmpSubjDir = fullfile(tempdir, ['subcort_rsa_' subName]);
        mkdir_p(tmpSubjDir);
        % Copy SPM.mat
        copyfile(spmMat, fullfile(tmpSubjDir, 'SPM.mat'));
        % Gunzip all .nii.gz to tmp
        for g = 1:numel(gzFiles)
            gunzip(fullfile(spmSubjDir, gzFiles(g).name), tmpSubjDir);
        end
        % Also copy any .nii files that already exist
        niiFiles = dir(fullfile(spmSubjDir, '*.nii'));
        for g = 1:numel(niiFiles)
            copyfile(fullfile(spmSubjDir, niiFiles(g).name), fullfile(tmpSubjDir, niiFiles(g).name));
        end
        % Update SPM.mat paths to point to tmp dir
        S = load(fullfile(tmpSubjDir, 'SPM.mat'));
        if isfield(S.SPM, 'swd')
            S.SPM.swd = tmpSubjDir;
        end
        % Update VY file paths: store just the filename (relative to swd)
        if isfield(S.SPM, 'Vbeta')
            for vi = 1:numel(S.SPM.Vbeta)
                [~, vname, vext] = fileparts(S.SPM.Vbeta(vi).fname);
                S.SPM.Vbeta(vi).fname = [vname vext];
            end
        end
        if isfield(S.SPM, 'VResMS') && ~isempty(S.SPM.VResMS)
            [~, vn, ve] = fileparts(S.SPM.VResMS.fname);
            S.SPM.VResMS.fname = [vn ve];
        end
        if isfield(S.SPM, 'xY') && isfield(S.SPM.xY, 'VY')
            for vi = 1:numel(S.SPM.xY.VY)
                [~, vn, ve] = fileparts(S.SPM.xY.VY(vi).fname);
                S.SPM.xY.VY(vi).fname = [vn ve];
            end
        end
        save(fullfile(tmpSubjDir, 'SPM.mat'), '-struct', 'S');
        spmMat = fullfile(tmpSubjDir, 'SPM.mat');
        fprintf('[INFO]   Decompressed betas to %s\n', tmpSubjDir);
    end

    % Load dataset from SPM model (beta estimates)
    ds = cosmo_fmri_dataset(spmMat);
    ds = cosmo_remove_useless_data(ds);
    cosmo_check_dataset(ds);

    % Clean up temporary decompressed files
    if ~isempty(tmpSubjDir) && exist(tmpSubjDir, 'dir')
        rmdir(tmpSubjDir, 's');
        fprintf('[INFO]   Cleaned up tmp dir for %s\n', subName);
    end

    if isempty(ds.samples)
        fprintf('[WARN]   Empty dataset for %s, skipping.\n', subName);
        continue;
    end

    % --------------------------------------------------------------------
    % Parse label strings to regressors and model vectors
    % --------------------------------------------------------------------
    [checkmateVec, strategyVec, stimVec, visSimilarityVec] = parse_label_regressors(ds);

    % Define regressor catalog
    regressors = struct();
    regressors.checkmate.targets = checkmateVec;
    regressors.strategy.targets = strategyVec;
    regressors.visual_similarity.targets = visSimilarityVec;

    % Stimulus-based targets for RSA averaging
    regressors.stimuli.targets = stimVec;

    % RSA model RDMs computed on first run (chunk==1)
    firstRunMask = (ds.sa.chunks == 1);
    regressors.checkmate.rdm = compute_rdm(checkmateVec(firstRunMask), 'similarity');
    regressors.strategy.rdm = compute_rdm(strategyVec(firstRunMask), 'similarity');
    regressors.visual_similarity.rdm = compute_rdm(visSimilarityVec(firstRunMask), 'similarity');

    % Prepare output directories
    subOutSVM = fullfile(outRootSVM, subName); mkdir_p(subOutSVM);
    subOutRSA = fullfile(outRootRSACorr, subName); mkdir_p(subOutRSA);

    % Target list (exclude stimuli helper)
    targetNames = fieldnames(regressors);
    targetNames = setdiff(targetNames, {'stimuli'});
    nReg = numel(targetNames);
    nROI = numel(region_ids);

    svm_mat = nan(nReg, nROI);
    rsa_mat = nan(nReg, nROI);

    % Measures
    classifier = @cosmo_classify_svm;
    rsa_measure = @cosmo_target_dsm_corr_measure;
    rsa_args = struct('center_data', true);

    % Map atlas labels to dataset feature space
    % ds.fa contains i,j,k voxel indices for each feature in the dataset
    feat_i = ds.fa.i(:);
    feat_j = ds.fa.j(:);
    feat_k = ds.fa.k(:);
    nFeatures = numel(feat_i);

    % Look up atlas label for each dataset feature
    feat_roi_labels = zeros(nFeatures, 1);
    for fi = 1:nFeatures
        feat_roi_labels(fi) = roi_data(feat_i(fi), feat_j(fi), feat_k(fi));
    end
    fprintf('[INFO]   Mapped %d features to atlas; %d in subcortical ROIs\n', ...
        nFeatures, sum(feat_roi_labels > 0));

    % ---------------------- ROI loop ----------------------
    for r = 1:nROI
        rid = region_ids(r);
        mask = (feat_roi_labels == rid);
        nVox = sum(mask);
        if nVox == 0
            fprintf('[WARN]   ROI %d (%s): no features overlap, skipping.\n', rid, region_names{r});
            continue;
        end

        ds_slice = cosmo_slice(ds, mask, 2);
        ds_slice = cosmo_remove_useless_data(ds_slice);
        if isempty(ds_slice.samples) || size(ds_slice.samples,2) < 6
            fprintf('[WARN]   ROI %d (%s): too few voxels (%d after cleanup), skipping.\n', ...
                rid, region_names{r}, size(ds_slice.samples,2));
            continue;
        end
        fprintf('[INFO]   ROI %d (%s): %d features\n', rid, region_names{r}, size(ds_slice.samples,2));

        % Build run-averaged dataset by stimulus identity for RSA
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
                    fprintf('[WARN]   SVM failed for %s, ROI %d (%s): %s\n', ...
                        tname, rid, region_names{r}, ME.message);
                end
            end

            % ---------- RSA correlation ----------
            try
                rsa_args.target_dsm = regressors.(tname).rdm;
                res = rsa_measure(ds_rsa_averaged, rsa_args);
                if ~isempty(res.samples)
                    rsa_mat(t, r) = res.samples;
                end
            catch ME
                fprintf('[WARN]   RSA failed for %s, ROI %d (%s): %s\n', ...
                    tname, rid, region_names{r}, ME.message);
            end
        end
    end

    % ---------------------- Save TSV outputs ----------------------
    % SVM decoding accuracy
    svm_tbl = array2table(svm_mat, 'VariableNames', matlab_safe_names(region_names));
    svm_tbl = addvars(svm_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    svmFilename = sprintf('%s_space-MNI152NLin2009cAsym_roi-cabnp_accuracy.tsv', subName);
    writetable(svm_tbl, fullfile(subOutSVM, svmFilename), 'FileType','text', 'Delimiter','\t');

    % RSA correlations
    rsa_tbl = array2table(rsa_mat, 'VariableNames', matlab_safe_names(region_names));
    rsa_tbl = addvars(rsa_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    rsaFilename = sprintf('%s_space-MNI152NLin2009cAsym_roi-cabnp_rdm.tsv', subName);
    writetable(rsa_tbl, fullfile(subOutRSA, rsaFilename), 'FileType','text', 'Delimiter','\t');

    fprintf('[INFO]   Saved SVM and RSA TSVs for %s\n', subName);
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
