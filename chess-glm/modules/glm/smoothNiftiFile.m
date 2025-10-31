function newFilePath = smoothNiftiFile(niiFile, outPath)
    % smoothNiftiFile - Smooth a .nii file with 6mm FWHM and save to temp.

    [niiFolder, niiName, niiExt] = fileparts(niiFile);
    subAndTask = split(niiName, "_"); %#ok<NASGU>

    if nargin < 2
        splitPath = strsplit(niiFile, '/');
        idx = find(contains(splitPath, 'BIDS'));
        bidsPath = strjoin(splitPath(1:idx), '/');
        outPath = fullfile(bidsPath, 'derivatives', 'SPM', 'smoothed');
    end

    if exist(outPath, 'dir') ~= 7, mkdir(outPath); end

    smoothFileName = strcat('smooth_', [niiName, niiExt]);
    newFilePath = fullfile(outPath, strrep(smoothFileName, niiExt, ['_smooth', niiExt]));

    if exist(newFilePath, 'file') == 2
        fprintf('SMOOTH: exists %s\n', newFilePath);
    else
        matlabbatch{1}.spm.spatial.smooth.data  = {niiFile};
        matlabbatch{1}.spm.spatial.smooth.fwhm  = [6 6 6];
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

