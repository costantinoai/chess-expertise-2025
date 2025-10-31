%% ========================================================================
%% First-level GLM in SPM (autocontrasts)
%%
%% Title
%% First-level SPM GLM with automatic contrast specification
%%
%% METHODS
%% Overview
%% Performs subject-level GLM estimation in SPM using fMRIPrep-preprocessed
%% BOLD series and BIDS events/confounds. Contrasts are specified
%% automatically via wildcard matching on design-matrix regressor names to
%% target the main study effects (Check vs No-Check; All vs Rest).
%%
%% Data
%% - Inputs:
%%   - fMRIPrep derivatives (preprocessed BOLD in MNI/T1w space; TSV/JSON
%%     confounds) under `<DERIVATIVES>/fmriprep`.
%%   - BIDS events TSVs under `<BIDS>/sub-XX/func`.
%% - Sample: all available subjects `sub-*` under fMRIPrep derivatives.
%%
%% Procedure
%% 1) Discover subjects with fMRIPrep outputs.
%% 2) For each subject and selected task:
%%    - Find runs, events TSVs, and confounds TSV/JSON.
%%    - Optional smoothing of the BOLD time series.
%%    - Specify GLM (HRF, high-pass, confounds regressors from pipeline).
%%    - Estimate model (SPM.mat).
%%    - Create T-contrasts via wildcard-based autocontrast mapping:
%%        • Check > No-Check: +1 on C* regressors, −1 on NC*
%%        • All > Rest: +1 on C* and NC* regressors (implicit baseline)
%%
%% Statistical Tests
%% - Subject-level first-level GLM estimation.
%% - T contrasts created as above; no second-level tests in this script.
%%
%% Outputs
%% - Per-subject SPM GLM directory:
%%     `<DERIVATIVES>/fmriprep-SPM_smoothed-<MM>_GS-FD-HMP_brainmasked/<SPACE>/fmriprep-SPM-<SPACE>/GLM/sub-XX/exp/`
%%   containing `SPM.mat`, `con_*.nii`, and associated files.
%% - A copy of the invoked helper script is stored in each subject’s GLM dir.
%%
%% Strict requirements
%% - Requires SPM on MATLAB path; fMRIPrep outputs must follow BIDS derivatives.
%% - No figures are generated here; plotting belongs to 91_* scripts.
%% - No silent fallbacks; missing files raise errors.
%%
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------
% Derivatives and BIDS roots (overridable via env)
defaultDeriv = '/data/projects/chess/data/BIDS/derivatives';
defaultBids  = '/data/projects/chess/data/BIDS';
DERIVATIVES  = getenv_default('CHESS_BIDS_DERIVATIVES', defaultDeriv);
BIDS_ROOT    = getenv_default('CHESS_BIDS_ROOT',        defaultBids);

% Space and smoothing
SPACE        = getenv_default('CHESS_GLM_SPACE', 'MNI');          % 'MNI' or 'T1w'
SMOOTH_MM    = str2double_safe(getenv_default('CHESS_GLM_SMOOTH_MM', '6')); % e.g., 6

% fMRIPrep root and output root
fmriprepRoot = fullfile(DERIVATIVES, 'fmriprep');
outRoot      = fullfile(DERIVATIVES, sprintf('fmriprep-SPM_smoothed-%d_GS-FD-HMP_brainmasked', SMOOTH_MM), ...
                        SPACE, sprintf('fmriprep-SPM-%s', SPACE));
tempDir      = fullfile(DERIVATIVES, 'fmriprep-preSPM');

% Subject selection and runs
selectedSubjectsList = '*';   % list of integers or '*'
selectedRuns         = '*';   % integer or '*'

% Task and autocontrasts (wildcard mapping)
selectedTasks(1).name = 'exp';
selectedTasks(1).contrasts = {'Check > No-Check', 'All > Rest'};
selectedTasks(1).weights(1) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', -1);
selectedTasks(1).weights(2) = struct('C_WILDCARD___WILDCARD_', 1, 'NC_WILDCARD___WILDCARD_', 1);
selectedTasks(1).smoothBool = true; % smooth inputs before GLM

% Confound regression pipeline
% Options include: HMP-[6|12|24], GS-[1|2|4], CSF_WM-[2|4|8], aCompCor-[10|50],
% MotionOutlier, Cosine, FD, Null
pipeline = {'HMP-6','FD','GS-1'};

% Thresholds were used to produce figures in the old implementation.
% We pass an empty list now to disable figure generation in run_subject_glm.
thresholds = {};  % no overlays in 01_*

fprintf('[INFO] fMRIPrep root: %s\n', fmriprepRoot);
fprintf('[INFO] BIDS root:     %s\n', BIDS_ROOT);
fprintf('[INFO] Output root:   %s\n', outRoot);

%% --------------------------- Paths for helpers ---------------------------
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'modules'));
addpath(fullfile(thisDir, 'modules', 'glm'));

%% --------------------------- Subject discovery ---------------------------
sub_paths = findSubjectsFolders(fmriprepRoot, selectedSubjectsList);
fprintf('[INFO] Found %d subject(s)\n', numel(sub_paths));

%% --------------------------- Main loop ----------------------------------
% Use parfor if Parallel Toolbox is available; otherwise fall back to for
canParallel = license('test','Distrib_Computing_Toolbox') == 1;

if canParallel
    parfor i = 1:length(sub_paths)
        run_subject_glm(sub_paths(i).folder, sub_paths(i).name, selectedTasks, selectedRuns, ...
            fmriprepRoot, BIDS_ROOT, outRoot, tempDir, pipeline, SPACE, thresholds);
    end
else
    for i = 1:length(sub_paths)
        run_subject_glm(sub_paths(i).folder, sub_paths(i).name, selectedTasks, selectedRuns, ...
            fmriprepRoot, BIDS_ROOT, outRoot, tempDir, pipeline, SPACE, thresholds);
    end
end

fprintf('\n[INFO] GLM autocontrast completed for all subjects.\n');

%% ========================================================================
%% Local helpers
%% ========================================================================
function out = getenv_default(name, default)
    val = getenv(name);
    if isempty(val), out = default; else, out = val; end
end

function v = str2double_safe(s)
    v = str2double(s);
    if isnan(v), v = 0; end
end

