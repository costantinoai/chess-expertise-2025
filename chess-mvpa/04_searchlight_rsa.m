%% ========================================================================
%% Whole-Brain Searchlight RSA (CoSMoMVPA)
%%
%% METHODS
%% =======
%%
%% Rationale
%% ---------
%% Whole-brain searchlight analysis extends ROI-based representational
%% similarity analysis (RSA) by computing local neural-model RDM correlations
%% in spherical neighborhoods centered on each voxel. This approach identifies
%% brain regions where multivoxel patterns align with theoretical model RDMs
%% without constraining the analysis to predefined anatomical regions. By
%% sliding a searchlight sphere across the brain, we can map the spatial
%% distribution of representational geometries corresponding to checkmate
%% status, visual similarity, and strategic category structure.
%%
%% Data
%% ----
%% Trial-wise beta estimates from unsmoothed first-level GLMs (40 stimuli:
%% 20 checkmate, 20 non-checkmate) were extracted for each of 40 participants
%% (20 experts, 20 novices). Beta images were averaged within each unique
%% stimulus across runs to produce one multivoxel pattern per stimulus per
%% subject (40 patterns × n_voxels). This averaging increases reliability
%% by collapsing across run-specific noise.
%%
%% Searchlight Procedure
%% ----------------------
%% For each subject:
%%   1. Load trial-wise beta images from SPM first-level GLM
%%   2. Parse condition labels to extract stimulus IDs and model regressors:
%%        - checkmate:          binary (checkmate vs non-checkmate)
%%        - visual_similarity:  20-class visual identity (ignoring strategy)
%%        - strategy:           multi-class strategic category
%%   3. Average beta patterns across runs for each unique stimulus (40 patterns)
%%   4. Define spherical searchlight neighborhoods (radius = 3 voxels ≈ 6 mm)
%%   5. For each voxel center:
%%        a. Extract multivoxel patterns from surrounding neighborhood
%%        b. Compute local neural RDM (40×40) using correlation distance (1-r)
%%        c. Correlate (Pearson) with each of three model RDMs
%%        d. Store correlation coefficient at center voxel
%%   6. Save whole-brain RSA maps (one per model) as NIfTI files
%%
%% Model RDMs reflect high-level conceptual dimensions (checkmate, strategy)
%% and lower-level perceptual features (visual similarity). Model RDMs for
%% categorical variables are constructed as binary dissimilarity matrices
%% (0 if same category, 1 if different).
%%
%% Neural RDMs are computed using correlation distance (1 - Pearson r)
%% between voxel patterns. Patterns are mean-centered before RDM computation
%% to remove baseline differences.
%%
%% Statistical Analysis
%% --------------------
%% Subject-level searchlight maps quantify local neural-model similarity at
%% each voxel. Group-level statistical testing (one-sample t-tests, two-sample
%% t-tests) is performed in downstream Python scripts using standard parametric
%% approaches with cluster-based correction or FDR.
%%
%% Computational Details
%% ---------------------
%% - Searchlight radius: 3 voxels (approximately 6 mm for 2mm isotropic data)
%% - Neighborhood size: approximately 123 voxels per sphere (4/3 * π * r³)
%% - Parallelization: CoSMoMVPA uses MATLAB's parallel computing (all cores)
%% - Distance metric: Correlation distance (1 - r) for neural RDMs
%% - RDM comparison: Pearson correlation between neural and model RDMs
%% - Mean centering: Applied to voxel patterns before computing neural RDMs
%%
%% Outputs
%% -------
%% Subject-level whole-brain RSA maps saved to:
%%   <BIDS_DERIVATIVES>/mvpa/<timestamp>_searchlight_rsa/sub-XX/
%%     ├── sub-XX_searchlight_checkmate.nii.gz
%%     ├── sub-XX_searchlight_visual_similarity.nii.gz
%%     └── sub-XX_searchlight_strategy.nii.gz
%%
%% Each map contains Pearson r values at each voxel, quantifying the strength
%% of neural-model correspondence. Values range from -1 (perfect anti-correlation)
%% to +1 (perfect correlation).
%%
%% Dependencies
%% ------------
%% - CoSMoMVPA toolbox (https://www.cosmomvpa.org/)
%% - SPM12 (for cosmo_fmri_dataset and cosmo_map2fmri)
%%
%% Notes
%% -----
%% This script performs subject-level analysis only. Group-level statistics,
%% visualization, and table generation are performed by downstream Python
%% scripts following the repository's separation of concerns (01_* = analysis
%% artifacts, 81_* = tables, 91_* = figures).
%%
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------

% BIDS derivatives root (override with env CHESS_BIDS_DERIVATIVES)
defaultDeriv = '/data/projects/chess/data/BIDS/derivatives';
derivativesDir = getenv_default('CHESS_BIDS_DERIVATIVES', defaultDeriv);

% GLM root for unsmoothed SPM outputs (subject folders live here)
% Searchlight RSA uses unsmoothed data to preserve spatial specificity
glmRoot = fullfile(derivativesDir, 'fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked', 'MNI', 'fmriprep-SPM-MNI', 'GLM');

% Output root with timestamp for reproducibility
ts = datestr(now, 'yyyymmdd-HHMMSS');
outRoot = fullfile(derivativesDir, 'mvpa', [ts, '_searchlight_rsa']);
mkdir_p(outRoot);

fprintf('[INFO] Searchlight RSA analysis starting at: %s\n', datestr(now));
fprintf('[INFO] Outputs will be written under: %s\n', outRoot);

%% --------------------------- Subject discovery ---------------------------

subDirs = find_subjects(glmRoot, 'sub-*');
fprintf('[INFO] Found %d subject(s) under: %s\n\n', numel(subDirs), glmRoot);

%% ========================= MAIN SUBJECT LOOP =============================

for s = 1:numel(subDirs)
    subName = subDirs(s).name;  % e.g., 'sub-01'
    fprintf('\n[INFO] ========== Processing %s (subject %d/%d) ==========\n', ...
            subName, s, numel(subDirs));

    spmSubjDir = fullfile(glmRoot, subName, 'exp');
    spmMat = fullfile(spmSubjDir, 'SPM.mat');
    if ~exist(spmMat, 'file')
        fprintf('[WARN]   Missing SPM.mat for %s, skipping.\n', subName);
        continue;
    end

    %% 1) LOAD AND VALIDATE FMRI DATASET -----------------------------------

    fprintf('[INFO]   Loading dataset from SPM.mat...\n');
    ds = cosmo_fmri_dataset(spmMat);
    ds = cosmo_remove_useless_data(ds);
    cosmo_check_dataset(ds);

    if isempty(ds.samples)
        fprintf('[WARN]   Empty dataset for %s, skipping.\n', subName);
        continue;
    end

    fprintf('[INFO]   Dataset loaded: %d samples × %d voxels\n', ...
            size(ds.samples, 1), size(ds.samples, 2));

    %% 2) PARSE LABEL VECTORS FOR REGRESSORS -------------------------------

    % Parse condition labels to extract stimulus IDs and model regressors:
    %   - stimuliVec:      unique integer per stimulus (1-40)
    %   - checkmateVec:    binary (2=checkmate, 1=non-checkmate)
    %   - visualStimVec:   visual identity ignoring "(nomate)" suffix
    %   - categoriesVec:   strategic category (multi-class)
    fprintf('[INFO]   Parsing condition labels...\n');
    [stimVec, checkmateVec, visStimVec, categoriesVec] = parse_label_regressors(ds);

    %% 3) COMPUTE MODEL RDMs (40×40) ---------------------------------------

    % Model RDMs reflect theoretical dissimilarity structure based on
    % stimulus features. For categorical variables (checkmate, strategy),
    % RDMs are binary: 0 if same category, 1 if different category.
    % This captures category boundaries without assuming metric distances.
    fprintf('[INFO]   Computing model RDMs (40×40 binary dissimilarity)...\n');

    % Use first run samples to define RDM structure (all runs have same labels)
    firstRunMask = (ds.sa.chunks == 1);

    modelRDMs = struct();
    modelRDMs.checkmate      = compute_rdm(checkmateVec(firstRunMask),  'similarity');
    modelRDMs.visualStimuli  = compute_rdm(visStimVec(firstRunMask),    'similarity');
    modelRDMs.categories     = compute_rdm(categoriesVec(firstRunMask), 'similarity');

    %% 4) AVERAGE PATTERNS ACROSS RUNS PER STIMULUS -----------------------

    % Assign targets (stimulus IDs) to dataset
    ds.sa.targets = stimVec(:);

    % Average voxel patterns across runs for each unique stimulus
    % This yields one beta pattern per stimulus (40 patterns × n_voxels)
    % Averaging increases reliability by collapsing run-specific noise
    fprintf('[INFO]   Averaging beta patterns across runs (one pattern per stimulus)...\n');
    ds_avg = cosmo_fx(ds, @(x) mean(x, 1), 'targets');
    cosmo_check_dataset(ds_avg);

    fprintf('[INFO]   Averaged dataset: %d stimuli × %d voxels\n', ...
            size(ds_avg.samples, 1), size(ds_avg.samples, 2));

    %% 5) DEFINE SPHERICAL SEARCHLIGHT NEIGHBORHOOD -----------------------

    % Define spherical neighborhoods centered on each voxel
    % Radius = 3 voxels ≈ 6 mm (for 2mm isotropic MNI data)
    % Each sphere contains ~123 voxels on average (4/3 * π * r³)
    fprintf('[INFO]   Defining searchlight neighborhoods (radius=3 voxels)...\n');

    radius_vox = 3;  % voxels
    nh = cosmo_spherical_neighborhood(ds_avg, 'radius', radius_vox);

    avgVoxCount = mean(cellfun(@numel, nh.neighbors));
    fprintf('[INFO]   Searchlight defined: avg %.1f voxels per sphere\n', avgVoxCount);

    %% 6) RUN SEARCHLIGHT RSA FOR EACH MODEL RDM --------------------------

    % Map model field names to output-friendly suffixes
    suffixMap = containers.Map( ...
        {'checkmate', 'visualStimuli', 'categories'}, ...
        {'checkmate', 'visual_similarity', 'strategy'} ...
    );

    % Set parallel processing options (use all available cores)
    nCores = feature('numcores');
    optRSA = struct('nproc', nCores, 'progress', true);

    fprintf('[INFO]   Running searchlight RSA (parallelized on %d cores)...\n', nCores);

    % Create subject output directory
    subOutDir = fullfile(outRoot, subName);
    mkdir_p(subOutDir);

    % Loop over each model RDM and run searchlight RSA
    for key = {'checkmate', 'visualStimuli', 'categories'}
        regName = key{1};
        suffix = suffixMap(regName);

        fprintf('[INFO]   ... %s\n', suffix);

        % Define RSA measure arguments
        rsa_args = struct();
        rsa_args.target_dsm  = modelRDMs.(regName);  % 40×40 model RDM
        rsa_args.center_data = true;                 % mean-center patterns before correlation

        % Run searchlight RSA
        % At each voxel center:
        %   1. Extract neighborhood patterns (n_stimuli × n_voxels_in_sphere)
        %   2. Compute local neural RDM using correlation distance (1-r)
        %   3. Correlate (Pearson) with model RDM
        %   4. Store correlation coefficient at center voxel
        tic;
        sl_rsa = cosmo_searchlight(ds_avg, nh, @cosmo_target_dsm_corr_measure, ...
                                   rsa_args, optRSA);
        elapsed = toc;

        % Save NIfTI result (gzipped to save space)
        outFile = fullfile(subOutDir, sprintf('%s_searchlight_%s.nii.gz', ...
                                              subName, suffix));
        cosmo_map2fmri(sl_rsa, outFile);

        fprintf('[INFO]     Saved: %s (%.1f sec)\n', outFile, elapsed);
    end

    fprintf('[INFO]   Completed %s\n', subName);
end

fprintf('\n[INFO] ========== All subjects complete ==========\n');
fprintf('[INFO] Searchlight RSA finished at: %s\n', datestr(now));
fprintf('[INFO] Results saved to: %s\n', outRoot);

%% ========================================================================
%  HELPER FUNCTIONS (consistent with 01_roi_mvpa_main.m)
%% ========================================================================

function out = getenv_default(name, default)
    % Get environment variable with fallback default
    val = getenv(name);
    if isempty(val)
        out = default;
    else
        out = val;
    end
end

function mkdir_p(p)
    % Create directory if it doesn't exist (like mkdir -p)
    if ~exist(p, 'dir')
        mkdir(p);
    end
end

function d = find_subjects(root, pattern)
    % Find all subject directories matching pattern
    d = dir(fullfile(root, pattern));
    d = d([d.isdir]);
end

function [checkmateVec, categoriesVec, stimVec, visStimVec] = parse_label_regressors(ds)
    % Parse label strings to extract regressors
    % Matches logic from 01_roi_mvpa_main.m for consistency
    %
    % Returns:
    %   checkmateVec  : binary (2=checkmate 'C', 1=non-checkmate 'NC')
    %   categoriesVec : dense integer codes for strategic categories
    %   stimVec       : dense integer codes for unique stimuli (1-40)
    %   visStimVec    : dense integer codes for visual identity (ignoring strategy)

    labels = ds.sa.labels(:);

    % Checkmate C/NC → 2 levels encoded as 2/1
    cmLabels = regexp(labels, '(?<=\s)(C|NC)\d+', 'match', 'once');
    checkmateVec = cellfun(@(x) strcmpi(x(1), 'C') + 1, cmLabels);

    % Categories: concatenated (C|NC)(\d+) mapped to dense integers
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
    % Compute dissimilarity matrix from vector of category labels
    %
    % Parameters:
    %   vec    : n×1 vector of category labels (integers)
    %   metric : 'similarity' for binary dissimilarity (0=same, 1=different)
    %
    % Returns:
    %   RDM : n×n symmetric dissimilarity matrix

    v = vec(:);
    switch lower(metric)
        case 'similarity'
            % Binary dissimilarity: 0 if same category, 1 if different
            RDM = double(bsxfun(@ne, v, v'));
        otherwise
            error('Unsupported RDM metric: %s', metric);
    end
end
