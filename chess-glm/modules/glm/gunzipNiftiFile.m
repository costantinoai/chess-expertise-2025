function gunzippedNii = gunzipNiftiFile(niiGzFile, outPath)
    % gunzipNiftiFile - Decompress .nii.gz into a temp derivatives location.

    [~, niiGzName, niiGzExt] = fileparts(niiGzFile);
    nameSplits = split(niiGzName, "_");
    selectedSub = nameSplits{1}; %#ok<NASGU>

    if nargin < 2
        splitPath = strsplit(niiGzFile, '/');
        idx = find(contains(splitPath, 'BIDS'));
        bidsPath = strjoin(splitPath(1:idx), '/');
        outPath = fullfile(bidsPath, 'derivatives', 'fmriprep-preSPM', 'gunzipped', selectedSub);
    end

    if exist(outPath, 'dir') ~= 7, mkdir(outPath); end

    newFilePath = fullfile(outPath, niiGzName);
    if exist(newFilePath, 'file') == 2
        fprintf('GUNZIP: exists %s\n', newFilePath);
        gunzippedNii = {newFilePath};
    else
        fprintf('GUNZIP: decompressing %s ...\n', [niiGzName, niiGzExt])
        gunzippedNii = gunzip(niiGzFile, outPath);
        fprintf('GUNZIP: written %s\n', newFilePath);
    end
end

