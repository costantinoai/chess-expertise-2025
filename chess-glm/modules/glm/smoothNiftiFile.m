function newFilePath = smoothNiftiFile(niiFile, outPath, fwhm)
    % smoothNiftiFile - Smooth a .nii file with given FWHM and save to temp.

    [niiFolder, niiName, niiExt] = fileparts(niiFile);
    subAndTask = split(niiName, "_"); %#ok<NASGU>

    if nargin < 2 || isempty(outPath)
        % Fallback workdir for smoothed intermediate BOLD files. All callers
        % in the current pipeline pass an explicit outPath, so this branch is
        % effectively dead code retained for API compatibility.
        splitPath = strsplit(niiFile, '/');
        idx = find(contains(splitPath, 'BIDS'));
        bidsPath = strjoin(splitPath(1:idx), '/');
        outPath = fullfile(bidsPath, 'derivatives', 'fmriprep_spm-smoothed', 'workdir');
    end
    if nargin < 3 || isempty(fwhm)
        fwhm = [4 4 4]; % Default to 4 mm per manuscript
    end

    if exist(outPath, 'dir') ~= 7, mkdir(outPath); end

    smoothFileName = strcat('smooth_', [niiName, niiExt]);
    newFilePath = fullfile(outPath, strrep(smoothFileName, niiExt, ['_smooth', niiExt]));

    if exist(newFilePath, 'file') == 2
        fprintf('SMOOTH: exists %s\n', newFilePath);
    else
        matlabbatch{1}.spm.spatial.smooth.data  = {niiFile};
        matlabbatch{1}.spm.spatial.smooth.fwhm  = fwhm;
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im    = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix= 'smooth_';

        spm_jobman('initcfg'); spm('defaults','fmri');
        fprintf('SMOOTH: smoothing %s ...\n', [niiName, niiExt])
        evalc('spm_jobman(''run'', matlabbatch);');
        movefile(fullfile(niiFolder, smoothFileName), newFilePath);
        fprintf('SMOOTH: written %s\n', newFilePath);
    end
end
