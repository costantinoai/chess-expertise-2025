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
    cfg.sourcedata     = fullfile(cfg.bidsRoot, 'sourcedata');
    cfg.participants   = fullfile(cfg.bidsRoot, 'participants.tsv');
    cfg.stimuliFile    = fullfile(cfg.bidsRoot, 'stimuli', 'stimuli.tsv');

    % ======================================================================
    % ROI atlases (under BIDS/sourcedata/atlases/ -- primary reference
    % atlases are stored in sourcedata per the BIDS Templates and Atlases
    % spec).
    % ======================================================================
    cfg.atlasRoot = fullfile(cfg.sourcedata, 'atlases');

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

    % ======================================================================
    % First-level SPM GLMs (provenance-chain naming)
    % ======================================================================
    cfg.spmUnsmoothed  = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed');
    cfg.spmSmoothed    = fullfile(cfg.derivatives, 'fmriprep_spm-smoothed');

    % ======================================================================
    % Analysis derivatives (produced from spmUnsmoothed unless noted)
    % ======================================================================
    cfg.mvpaRsa                 = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_rsa');
    cfg.mvpaDecoding            = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_decoding');
    cfg.rsaSearchlight          = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_searchlight-rsa');
    cfg.mvpaRsaRunMatched       = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_rsa-run-matched');
    cfg.mvpaRsaSubcortical      = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_rsa-subcortical');
    cfg.mvpaDecodingSubcortical = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_decoding-subcortical');
    cfg.manifold                = fullfile(cfg.derivatives, 'fmriprep_spm-unsmoothed_manifold');
    cfg.behavioralRsa           = fullfile(cfg.derivatives, 'behavioral-rsa');
    cfg.eyeTracking             = fullfile(cfg.derivatives, 'bidsmreye');

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
