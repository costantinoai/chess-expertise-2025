function cfg = chess_config()
% CHESS_CONFIG Central path configuration for all MATLAB analysis scripts.
%
%   cfg = chess_config()
%
%   Returns a struct with all paths used across the pipeline. This is the
%   MATLAB equivalent of common/constants.py -- edit paths here, not in
%   individual scripts.
%
%   All paths can be overridden via environment variables (see inline comments).
%   The env-var mechanism allows running on different machines without editing
%   this file.

    % ======================================================================
    % External data root
    % ======================================================================
    cfg.dataRoot = getenv_or('/media/costantino_ai/eik-T9/manuscript-data', ...
                             'CHESS_DATA_ROOT');

    % ======================================================================
    % BIDS structure
    % ======================================================================
    cfg.bidsRoot       = fullfile(cfg.dataRoot, 'BIDS');
    cfg.derivatives    = fullfile(cfg.bidsRoot, 'derivatives');
    cfg.participants   = fullfile(cfg.bidsRoot, 'participants.tsv');
    cfg.stimuliFile    = fullfile(cfg.bidsRoot, 'stimuli', 'stimuli.tsv');

    % ======================================================================
    % ROI atlases (under BIDS/derivatives/atlases/)
    % ======================================================================
    cfg.atlasRoot = fullfile(cfg.derivatives, 'atlases');

    % Glasser 22 bilateral cortical ROIs
    cfg.roiGlasser22Dir   = fullfile(cfg.atlasRoot, 'glasser22');
    cfg.roiGlasser22Atlas = getenv_or( ...
        fullfile(cfg.roiGlasser22Dir, ...
            'tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii'), ...
        'CHESS_ROI_ATLAS_22');
    cfg.roiGlasser22TSV = getenv_or( ...
        fullfile(cfg.roiGlasser22Dir, 'region_info.tsv'), ...
        'CHESS_ROI_TSV_22');

    % Glasser 180 bilateral cortical ROIs
    cfg.roiGlasser180Dir = fullfile(cfg.atlasRoot, 'glasser180');

    % CAB-NP subcortical ROIs
    cfg.roiCabnpDir   = fullfile(cfg.atlasRoot, 'cab-np');
    cfg.roiCabnpAtlas = getenv_or( ...
        fullfile(cfg.roiCabnpDir, ...
            'tpl-MNI152NLin2009cAsym_res-02_atlas-CABNP_desc-subcortical_bilateral_resampled.nii'), ...
        'CHESS_ROI_ATLAS_CABNP');
    cfg.roiCabnpTSV = getenv_or( ...
        fullfile(cfg.roiCabnpDir, 'region_info.tsv'), ...
        'CHESS_ROI_TSV_CABNP');

    % ======================================================================
    % Preprocessing derivatives
    % ======================================================================
    cfg.fmriprep       = fullfile(cfg.derivatives, 'fmriprep');
    cfg.spmDir         = fullfile(cfg.derivatives, 'SPM');
    cfg.glmUnsmoothed  = fullfile(cfg.spmDir, 'GLM-unsmoothed');
    cfg.glmSmooth4     = fullfile(cfg.spmDir, 'GLM-smooth4');

    % ======================================================================
    % Analysis derivatives
    % ======================================================================
    cfg.mvpaRsa              = fullfile(cfg.derivatives, 'mvpa-rsa');
    cfg.mvpaDecoding         = fullfile(cfg.derivatives, 'mvpa-decoding');
    cfg.rsaSearchlight       = fullfile(cfg.derivatives, 'rsa_searchlight');
    cfg.mvpaRsaSubcortical   = fullfile(cfg.derivatives, 'mvpa-rsa-subcortical');
    cfg.mvpaDecodingSubcortical = fullfile(cfg.derivatives, 'mvpa-decoding-subcortical');
    cfg.eyeTracking          = fullfile(cfg.derivatives, 'eye-tracking');

end

% =========================================================================
% Helper: return env var value if set, otherwise use default
% =========================================================================
function out = getenv_or(default, envName)
    val = getenv(envName);
    if isempty(val)
        out = default;
    else
        out = val;
    end
end
