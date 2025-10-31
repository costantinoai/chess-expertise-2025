%% ========================================================================
%% Within-Group Second-Level (SPM, multiple contrasts)
%%
%% Title
%% Within-group (1-sample t-test) analyses for Experts and Novices
%%
%% METHODS
%% Overview
%% For each first-level contrast image (e.g., con_0001.nii), performs a
%% separate one-sample t-test within each group (experts, novices). Optionally
%% applies Gaussian smoothing to the contrast images before model estimation.
%%
%% Data
%% - Inputs: first-level subject GLM directories containing `con_*.nii` files
%%   produced by 01_spm_glm_autocontrast.m under:
%%     <DERIVATIVES>/fmriprep-SPM_smoothed-<MM>_GS-FD-HMP_brainmasked/<SPACE>/fmriprep-SPM-<SPACE>/GLM/sub-XX/exp/
%% - Group membership: parsed from `<BIDS_ROOT>/participants.tsv` (column `group`).
%%
%% Procedure
%% 1) Load group membership (experts, novices) from participants.tsv.
%% 2) For each contrast file in `CONTRAST_FILES`:
%%    a) Collect subject contrast images for the group; smooth if requested.
%%    b) Specify factorial design (1-sample t-test) and estimate the model.
%%    c) Define T-contrast [1] (Group mean > 0).
%%
%% Statistical Tests
%% - One-sample t-tests within experts and within novices, per contrast.
%%
%% Outputs
%% - For each contrast and group, a second-level folder under the GLM root:
%%     2ndLevel_Experts_<cbase>[/smoothed]
%%     2ndLevel_NonExperts_<cbase>[/smoothed]
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
SMOOTH_MM    = str2double_safe(getenv_default('CHESS_GLM_SMOOTH_MM', '4'));

% First-level GLM root (BIDS-like): derivatives/SPM/smooth<MM>
glmRoot = fullfile(DERIVATIVES, 'SPM', sprintf('smooth%d', SMOOTH_MM));

% No second-level smoothing; group analyses run on first-level smoothed (4mm) contrasts

% Contrasts to process
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

    % Output folders
    outExp = fullfile(glmRoot, 'group', ['Experts_' cbase]);
    outNov = fullfile(glmRoot, 'group', ['NonExperts_' cbase]);
    if ~exist(outExp,'dir'), mkdir(outExp); end
    if ~exist(outNov,'dir'), mkdir(outNov); end

    %% Experts 1-sample t-test
    expertScans = collect_contrast_paths(glmRoot, experts, contrastFile, false);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {outExp};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = expertScans;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.ancova = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.dept = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outExp, 'SPM.mat')};

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(outExp, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Group Mean (Experts): ' cbase];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 0;

    spm('Defaults','fMRI'); spm_jobman('initcfg'); spm_jobman('run', matlabbatch);
    fprintf('[INFO] Experts 1-sample done for %s\n', cbase);

    %% Novices 1-sample t-test
    noviceScans = collect_contrast_paths(glmRoot, novices, contrastFile, false);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {outNov};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = noviceScans;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.ancova = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.t1.dept = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outNov, 'SPM.mat')};

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(outNov, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = ['Group Mean (Non-Experts): ' cbase];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 0;

    spm('Defaults','fMRI'); spm_jobman('initcfg'); spm_jobman('run', matlabbatch);
    fprintf('[INFO] Non-Experts 1-sample done for %s\n', cbase);
end

fprintf('\n[INFO] Within-group analyses complete.\n');

%% ========================================================================
function out = getenv_default(name, default)
    val = getenv(name);
    if isempty(val), out = default; else, out = val; end
end

function v = str2double_safe(s)
    v = str2double(s);
    if isnan(v), v = 0; end
end
