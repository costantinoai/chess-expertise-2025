%% ========================================================================
%% ROI-based Decoding and RSA (CoSMoMVPA) — Fine Dimensions (Checkmate-only)
%%
%% Purpose
%% - Per-subject ROI decoding (SVM) and ROI RSA correlations for 5 finer
%%   dimensions, using only the 20 checkmate boards:
%%     1) strategy_cm         (categorical)      → 'similarity'
%%     2) motif               (categorical)      → 'similarity'
%%     3) pieces_total        (ordinal counts)   → 'subtraction'
%%     4) legal_moves         (ordinal counts)   → 'subtraction'
%%     5) moves_to_mate       (ordinal counts)   → 'subtraction'
%%
%% Output
%% - Subject-level TSV files saved under:
%%     <BIDS_DERIVATIVES>/mvpa/<timestamp>_glasser_regions_bilateral_fine/
%%       ├── svm/sub-XX/mvpa_cv.tsv          (decoding accuracy per ROI per target)
%%       └── rsa_corr/sub-XX/rsa_corr.tsv    (RSA r-values per ROI per target)
%%
%% Notes
%% - Analysis-only (no figures). Plotting/reporting are in Python.
%% - CoSMoMVPA + SPM required on MATLAB path.
%% - Stimulus labels for fine targets loaded from stimuli.tsv
%%   (override path via CHESS_STIMULI_TSV).
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------
defaultDeriv = '/data/projects/chess/data/BIDS/derivatives';
derivativesDir = getenv_default('CHESS_BIDS_DERIVATIVES', defaultDeriv);

glmRoot = fullfile(derivativesDir, 'fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked', 'MNI', 'fmriprep-SPM-MNI', 'GLM');

roiAtlas = getenv_default('CHESS_ROI_ATLAS_22', ...
    fullfile(derivativesDir, 'rois', 'glasser22', 'tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii'));
roiTSV = getenv_default('CHESS_ROI_TSV_22', ...
    fullfile(derivativesDir, 'rois', 'glasser22', 'region_info.tsv'));

stimuliTSV = getenv_default('CHESS_STIMULI_TSV', '/media/costantino_ai/eik-T9/manuscript-data/stimuli/stimuli.tsv');

ts = datestr(now, 'yyyymmdd-HHMMSS');
outRoot = fullfile(derivativesDir, 'mvpa', [ts, '_glasser_regions_bilateral_fine']);
outRootSVM = fullfile(outRoot, 'svm');
outRootRSACorr = fullfile(outRoot, 'rsa_corr');
mkdir_p(outRootSVM); mkdir_p(outRootRSACorr);

fprintf('[INFO] Outputs will be written under: %s\n', outRoot);

%% --------------------------- Subject discovery ---------------------------
subDirs = find_subjects(glmRoot, 'sub-*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmRoot);

%% --------------------------- Load ROIs -----------------------------------
[~, roi_data] = spm_read_vols(spm_vol(roiAtlas));
roiInfo = readtable(roiTSV, 'FileType','text', 'Delimiter','\t');
assert(ismember('index', roiInfo.Properties.VariableNames), 'ROI TSV must contain an index column');
region_ids = roiInfo.index(:)';
region_names = roiInfo.name(:)';

%% --------------------------- Load stimuli TSV ----------------------------
stimT = readtable(stimuliTSV, 'FileType','text', 'Delimiter','\t');

% Determine a canonical stimulus key column for joining
stimKeyCol = pick_first_present(stimT, {'stimulus_key','stimulus','label'});
assert(~isempty(stimKeyCol), 'Stimuli TSV must have a stimulus identifier column');

% Normalize key: lower-case, remove '(nomate)'
stimT.('stimulus_key_norm') = lower(erase(string(stimT.(stimKeyCol)), '(nomate)'));

% Map canonical target names → candidate TSV columns
targetColMap = struct();
targetColMap.strategy_cm  = pick_first_present(stimT, {'strategy_cm','strategy','categories'});
targetColMap.motif        = pick_first_present(stimT, {'motif','tactical_motif'});
targetColMap.pieces_total = pick_first_present(stimT, {'pieces_total','total_pieces'});
targetColMap.legal_moves  = pick_first_present(stimT, {'legal_moves','total_legal_moves'});
targetColMap.moves_to_mate= pick_first_present(stimT, {'moves_to_mate','white_moves_to_mate'});

% Validate that all necessary columns exist
fields = fieldnames(targetColMap);
for i=1:numel(fields)
    col = targetColMap.(fields{i});
    assert(~isempty(col), 'Missing column in stimuli TSV for target: %s', fields{i});
end

%% ========================= Main subject loop =============================
for s = 1:numel(subDirs)
    subName = subDirs(s).name;  % 'sub-XX'
    fprintf('\n[INFO] Processing %s\n', subName);

    spmSubjDir = fullfile(glmRoot, subName, 'exp');
    spmMat = fullfile(spmSubjDir, 'SPM.mat');
    if ~exist(spmMat, 'file')
        fprintf('[WARN]   Missing SPM.mat for %s, skipping.\n', subName);
        continue;
    end

    % Load full dataset
    ds = cosmo_fmri_dataset(spmMat);
    ds = cosmo_remove_useless_data(ds);
    cosmo_check_dataset(ds);
    if isempty(ds.samples), fprintf('[WARN]   Empty ds for %s, skipping.\n', subName); continue; end

    % Parse labels to derive sample properties
    [checkmateVec, ~, stimVec, visStimVec, lowerStim] = parse_label_regressors_full(ds);

    % Select only checkmate samples (value 2 in checkmateVec)
    cmMask = (checkmateVec == 2);
    if ~any(cmMask), fprintf('[WARN]   No checkmate samples for %s, skipping.\n', subName); continue; end
    ds_half = cosmo_slice(ds, cmMask, 1);
    ds_half = cosmo_remove_useless_data(ds_half);

    % Build normalized stimulus keys for the subset
    lowerStim_half = lower(erase(string(lowerStim(cmMask)), '(nomate)'));

    % Build fine target vectors aligned to ds_half samples using stimuli TSV
    fineTargets = struct();
    fineList = {'strategy_cm','motif','pieces_total','legal_moves','moves_to_mate'};
    for i=1:numel(fineList)
        key = fineList{i};
        col = targetColMap.(key);
        fineTargets.(key) = zeros(numel(lowerStim_half),1);
        for k = 1:numel(lowerStim_half)
            rows = strcmp(stimT.stimulus_key_norm, lowerStim_half(k));
            assert(any(rows), 'Stimulus key not found in TSV: %s', lowerStim_half(k));
            fineTargets.(key)(k) = stimT.(col)(find(rows,1));
        end
    end

    % Prepare helper 'stimuli_half' identity vector (for RSA averaging)
    stimuli_half = grp2idx(lowerStim_half); % dense 1..20

    % Compute model RDMs on first run of ds_half
    firstRunMask = (ds_half.sa.chunks == min(ds_half.sa.chunks));
    modelRDMs = struct();
    modelRDMs.strategy_cm   = compute_rdm(fineTargets.strategy_cm(firstRunMask), 'similarity');
    modelRDMs.motif         = compute_rdm(fineTargets.motif(firstRunMask), 'similarity');
    modelRDMs.pieces_total  = compute_rdm(fineTargets.pieces_total(firstRunMask), 'subtraction');
    modelRDMs.legal_moves   = compute_rdm(fineTargets.legal_moves(firstRunMask), 'subtraction');
    modelRDMs.moves_to_mate = compute_rdm(fineTargets.moves_to_mate(firstRunMask), 'subtraction');

    % Prepare outputs per subject
    subOutSVM = fullfile(outRootSVM, subName); mkdir_p(subOutSVM);
    subOutRSA = fullfile(outRootRSACorr, subName); mkdir_p(subOutRSA);

    % Initialize result mats (targets × rois)
    targetNames = fineList;
    nReg = numel(targetNames);
    nROI = numel(region_ids);
    svm_mat = nan(nReg, nROI);
    rsa_mat = nan(nReg, nROI);

    classifier = @cosmo_classify_svm;
    rsa_measure = @cosmo_target_dsm_corr_measure;
    rsa_args = struct('center_data', true);

    % ROI loop
    for r = 1:nROI
        rid = region_ids(r);
        mask = (roi_data == rid);
        if ~any(mask(:)), continue; end

        ds_slice = cosmo_slice(ds_half, mask, 2);
        ds_slice = cosmo_remove_useless_data(ds_slice);
        if isempty(ds_slice.samples) || size(ds_slice.samples,2) < 6, continue; end

        % Create RSA averaged slice by stimulus identity within checkmates
        ds_rsa = ds_slice;
        ds_rsa.sa.targets = stimuli_half;
        ds_rsa_averaged = cosmo_fx(ds_rsa, @(x) mean(x,1), 'targets');

        for t = 1:nReg
            tname = targetNames{t};

            % Decoding: assign targets for this fine dimension
            ds_dec = ds_slice;
            ds_dec.sa.targets = fineTargets.(tname);
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

            % RSA correlation: use corresponding model RDM
            try
                rsa_args.target_dsm = modelRDMs.(tname);
                res = rsa_measure(ds_rsa_averaged, rsa_args);
                if ~isempty(res.samples)
                    rsa_mat(t, r) = res.samples;
                end
            catch ME
                fprintf('[WARN]   RSA failed for %s, ROI %d: %s\n', tname, rid, ME.message);
            end
        end
    end

    % Save TSVs
    svm_tbl = array2table(svm_mat, 'VariableNames', matlab_safe_names(region_names));
    svm_tbl = addvars(svm_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    writetable(svm_tbl, fullfile(subOutSVM, 'mvpa_cv.tsv'), 'FileType','text', 'Delimiter','\t');

    rsa_tbl = array2table(rsa_mat, 'VariableNames', matlab_safe_names(region_names));
    rsa_tbl = addvars(rsa_tbl, string(targetNames), 'Before', 1, 'NewVariableNames','target');
    writetable(rsa_tbl, fullfile(subOutRSA, 'rsa_corr.tsv'), 'FileType','text', 'Delimiter','\t');

    fprintf('[INFO]   Saved SVM and RSA TSVs for %s\n', subName);
end

fprintf('\n[INFO] Done. Subject-level TSV files written to: %s\n', outRoot);

%% ========================================================================
%% Helper functions
%% ========================================================================

function out = getenv_default(name, default)
    val = getenv(name);
    if isempty(val), out = default; else, out = val; end
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

function col = pick_first_present(T, candidates)
    col = '';
    for i = 1:numel(candidates)
        if ismember(candidates{i}, T.Properties.VariableNames)
            col = candidates{i};
            return;
        end
    end
end

function [checkmateVec, categoriesVec, stimVec, visStimVec, lowerStim] = parse_label_regressors_full(ds)
    labels = ds.sa.labels(:);

    % Checkmate C/NC → 2 levels encoded as 2/1
    cmLabels = regexp(labels, '(?<=\s)(C|NC)\d+', 'match', 'once');
    checkmateVec = cellfun(@(x) strcmpi(x(1), 'C') + 1, cmLabels);

    % Categories (legacy multi-class across all 40)
    catTokens = regexp(labels, '(?<=\s)(C|NC)(\d+)', 'tokens', 'once');
    concatCats = cellfun(@(x) [x{1}, x{2}], catTokens, 'UniformOutput', false);
    uniqCats = unique(concatCats, 'stable');
    catMap = containers.Map(uniqCats, 1:numel(uniqCats));
    categoriesVec = cellfun(@(x) catMap(x), concatCats);

    % Stimulus strings
    stimLabels = regexp(labels, '(?<=_).*?(?=\*)', 'match', 'once');
    lowerStim = lower(stimLabels);
    uStim = unique(lowerStim, 'stable');
    stimMap = containers.Map(uStim, 1:numel(uStim));
    stimVec = cellfun(@(x) stimMap(x), lowerStim);

    % Visual identity ignoring '(nomate)'
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
        case 'subtraction'
            RDM = abs(bsxfun(@minus, v, v'));
        otherwise
            error('Unsupported RDM metric: %s', metric);
    end
end

