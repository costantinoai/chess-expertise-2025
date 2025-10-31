function scans = collect_contrast_paths(rootDir, subjectList, contrastFile, smoothBool, fwhm)
% COLLECT_CONTRAST_PATHS Collect paths to a given contrast across subjects.
% If smoothBool is true, smooth each contrast image using SPM and return the
% smoothed image paths with volume index ",1".

if nargin < 5
    fwhm = [6 6 6];
end

n = numel(subjectList);
in_paths = cell(n,1);
for i = 1:n
    subName = subjectList{i}; % e.g., 'sub-01'
    in_paths{i} = fullfile(rootDir, subName, 'exp', contrastFile);
    if ~exist(in_paths{i}, 'file')
        error('Contrast not found: %s', in_paths{i});
    end
end

if ~smoothBool
    scans = cellfun(@(p) [p ',1'], in_paths, 'UniformOutput', false);
    return;
end

% Smooth all at once using a single SPM batch
matlabbatch = {};
for i = 1:n
    matlabbatch{1}.spm.spatial.smooth.data{i,1} = in_paths{i}; %#ok<AGROW>
end
matlabbatch{1}.spm.spatial.smooth.fwhm   = fwhm;
matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
matlabbatch{1}.spm.spatial.smooth.im     = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = 's';

spm('Defaults','fMRI');
spm_jobman('initcfg');
spm_jobman('run', matlabbatch);

% Return smoothed file paths (prefix 's')
scans = cell(n,1);
for i = 1:n
    [p,f,e] = fileparts(in_paths{i});
    scans{i} = fullfile(p, ['s' f e ',1']);
end
end

