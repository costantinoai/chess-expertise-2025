function [experts, novices] = load_group_subjects(bidsRoot)
% LOAD_GROUP_SUBJECTS Read BIDS participants.tsv and return expert/novice lists.

participants_tsv = fullfile(bidsRoot, 'participants.tsv');
if ~exist(participants_tsv, 'file')
    error('participants.tsv not found at %s', participants_tsv);
end

tbl = readtable(participants_tsv, 'FileType','text', 'Delimiter','\t');

if ~ismember('participant_id', tbl.Properties.VariableNames) || ~ismember('group', tbl.Properties.VariableNames)
    error('participants.tsv must have columns participant_id and group');
end

experts = tbl.participant_id(strcmp(tbl.group,'expert'));
novices = tbl.participant_id(strcmp(tbl.group,'novice'));

% Ensure cellstr
if isstring(experts), experts = cellstr(experts); end
if isstring(novices), novices = cellstr(novices); end
end

