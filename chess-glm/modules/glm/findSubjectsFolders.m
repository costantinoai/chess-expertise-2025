function [filteredFolderStructure] = findSubjectsFolders(fmriprepRoot, selectedSubjectsList, excludedSubjectsList)
% FINDSUBJECTSFOLDERS Locate subject folders based on a list or wildcard.

sub_paths = dir(fullfile(fmriprepRoot, 'sub-*'));
sub_paths = sub_paths([sub_paths.isdir]);

if isnumeric(selectedSubjectsList) || (iscell(selectedSubjectsList) && ~isempty(selectedSubjectsList) && isnumeric(selectedSubjectsList{1}))
    % Normalize to numeric array
    if iscell(selectedSubjectsList), selectedSubjectsList = cell2mat(selectedSubjectsList); end
    subIDs = cellfun(@(x) sprintf('sub-%02d', x), num2cell(selectedSubjectsList), 'UniformOutput', false);
    sub_paths = sub_paths(ismember({sub_paths.name}, subIDs));
elseif ischar(selectedSubjectsList) && strcmp(selectedSubjectsList, '*')
    % keep all
else
    error('Invalid format for selectedSubjects. Use "*" or a list of integers.');
end

if nargin == 3
    excludedNames = cellfun(@(x) sprintf('sub-%02d', x), num2cell(excludedSubjectsList), 'UniformOutput', false);
    excludeMask = arrayfun(@(x) ismember(x.name, excludedNames), sub_paths);
    filteredFolderStructure = sub_paths(~excludeMask);
else
    filteredFolderStructure = sub_paths;
end
end

