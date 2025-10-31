%% ========================================================================
%% Two-Sample Second-Level (SPM, multiple contrasts)
%%
%% Title
%% Two-sample t-test (Experts vs Novices) for first-level contrasts
%%
%% METHODS
%% Overview
%% For each first-level contrast image (e.g., con_0001.nii), performs a
%% two-sample t-test comparing Experts vs Novices. Optionally applies
%% Gaussian smoothing to contrast images before model estimation.
%%
%% Data
%% - Inputs: first-level subject GLM directories containing `con_*.nii` files
%%   under the GLM root produced by 01_spm_glm_autocontrast.m.
%% - Group membership: parsed from `<BIDS_ROOT>/participants.tsv` (column `group`).
%%
%% Procedure
%% 1) Load group membership (experts, novices) from participants.tsv.
%% 2) For each contrast file in `CONTRAST_FILES`:
%%    a) Collect subject contrast images for both groups; smooth if requested.
%%    b) Specify factorial design (two-sample t-test) and estimate model.
%%    c) Define T-contrasts: Experts > Novices [1 -1] and Novices > Experts [-1 1].
%%
%% Statistical Tests
%% - Two-sample t-tests for each contrast.
%%
%% Outputs
%% - For each contrast, a second-level folder under the GLM root:
%%     2ndLevel_ExpVsNonExp_<cbase>[/smoothed]
%%   containing `SPM.mat` and group-level contrasts.
%%
%% Strict requirements
%% - No figures are generated here.
%% - Missing inputs raise errors; no silent fallbacks.
%%
%% ========================================================================

clear; clc;

%% --------------------------- Configuration -------------------------------
defaultDeriv = '/data/projects/chess/data/BIDS/derivatives';
defaultBids  = '/data/projects/chess/data/BIDS';
DERIVATIVES  = getenv_default('CHESS_BIDS_DERIVATIVES', defaultDeriv);
BIDS_ROOT    = getenv_default('CHESS_BIDS_ROOT',        defaultBids);
SPACE        = getenv_default('CHESS_GLM_SPACE', 'MNI');
SMOOTH_MM    = str2double_safe(getenv_default('CHESS_GLM_SMOOTH_MM', '6'));

glmRoot = fullfile(DERIVATIVES, sprintf('fmriprep-SPM_smoothed-%d_GS-FD-HMP_brainmasked', SMOOTH_MM), ...
                   SPACE, sprintf('fmriprep-SPM-%s', SPACE), 'GLM');

SMOOTH_SECOND_LEVEL = strcmp(getenv_default('CHESS_GLM_2ND_SMOOTH', '0'), '1');
SECOND_FWHM = [6 6 6];

CONTRAST_FILES = {'con_0001.nii', 'con_0002.nii'};

%% --------------------------- Paths for helpers ---------------------------
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'modules'));
addpath(fullfile(thisDir, 'modules', 'glm'));

%% --------------------------- Load groups ---------------------------------
[experts, novices] = load_group_subjects(BIDS_ROOT);
fprintf('[INFO] Experts: %d, Novices: %d\n', numel(experts), numel(novices));
fprintf('[INFO] GLM root: %s\n', glmRoot);

%% --------------------------- Loop contrasts ------------------------------
for c = 1:numel(CONTRAST_FILES)
    contrastFile = CONTRAST_FILES{c};
    [~, cbase, ~] = fileparts(contrastFile);

    outDir = fullfile(glmRoot, ['2ndLevel_ExpVsNonExp_' cbase]);
    if SMOOTH_SECOND_LEVEL
        outDir = [outDir filesep 'smoothed']; %#ok<AGROW>
    end
    if ~exist(outDir,'dir'), mkdir(outDir); end

    expScans = collect_contrast_paths(glmRoot, experts, contrastFile, SMOOTH_SECOND_LEVEL, SECOND_FWHM);
    novScans = collect_contrast_paths(glmRoot, novices, contrastFile, SMOOTH_SECOND_LEVEL, SECOND_FWHM);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {outDir};
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = expScans;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = novScans;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.dept     = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca    = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova   = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im         = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em         = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit     = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm    = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outDir, 'SPM.mat')};

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(outDir, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Experts > Non-Experts: ' cbase];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.name    = ['Non-Experts > Experts: ' cbase];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 0;

    spm('Defaults','fMRI'); spm_jobman('initcfg'); spm_jobman('run', matlabbatch);
    fprintf('[INFO] Two-sample done for %s\n', cbase);
end

fprintf('\n[INFO] Two-sample analyses complete.\n');

%% ========================================================================
function out = getenv_default(name, default)
    val = getenv(name);
    if isempty(val), out = default; else, out = val; end
end

function v = str2double_safe(s)
    v = str2double(s);
    if isnan(v), v = 0; end
end
