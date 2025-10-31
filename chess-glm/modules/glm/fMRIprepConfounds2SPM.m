function confounds = fMRIprepConfounds2SPM(json_path, tsv_path, pipeline)
% fMRIprepConfounds2SPM - Extracts and formats fMRI confounds for SPM analysis
%
% Copied from old-implementation with identical behavior. See header there
% for complete documentation of pipeline options.

% Read the TSV file containing the confound values
tsv_run = readtable(tsv_path, 'FileType', 'text');

% Open and read the JSON file, then parse it into a MATLAB structure
fid = fopen(json_path); raw = fread(fid, inf); str = char(raw'); fclose(fid);
json_run = jsondecode(str);

selected_keys = {};

% Early exit for Null
if any(strcmp(pipeline, 'Null'))
    confounds = table();
    return;
end

% HMP
if any(contains(pipeline, 'HMP'))
    idx = find(contains(pipeline, 'HMP'));
    conf_num_str = pipeline(idx(1));
    conf_num = str2double(strsplit(conf_num_str{1}, '-'){2});
    if ~any([6, 12, 24] == conf_num)
        error('HMP must be 6, 12, or 24.');
    else
        hmp_id = floor(conf_num / 6);
        if hmp_id > 0
            selected_keys = [selected_keys, {'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'}]; %#ok<AGROW>
        end
        if hmp_id > 1
            selected_keys = [selected_keys, {'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1', 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1'}]; %#ok<AGROW>
        end
        if hmp_id > 2
            selected_keys = [selected_keys, {'rot_x_power2', 'rot_y_power2', 'rot_z_power2', 'trans_x_power2', 'trans_y_power2', 'trans_z_power2', 'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2', 'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2'}]; %#ok<AGROW>
        end
    end
end

% GS
if any(contains(pipeline, 'GS'))
    idx = find(contains(pipeline, 'GS'));
    conf_num_str = pipeline(idx(1));
    conf_num = str2double(strsplit(conf_num_str{1}, '-'){2});
    if ~any([1, 2, 4] == conf_num)
        error('GS must be 1, 2, or 4.');
    else
        gs_id = conf_num;
        if gs_id > 0, selected_keys = [selected_keys, {'global_signal'}]; end %#ok<AGROW>
        if gs_id > 1, selected_keys = [selected_keys, {'global_signal_derivative1'}]; end %#ok<AGROW>
        if gs_id > 2, selected_keys = [selected_keys, {'global_signal_derivative1_power2', 'global_signal_power2'}]; end %#ok<AGROW>
    end
end

% CSF_WM
if any(contains(pipeline, 'CSF_WM'))
    idx = find(contains(pipeline, 'CSF_WM'));
    conf_num_str = pipeline(idx(1));
    conf_num = str2double(strsplit(conf_num_str{1}, '-'){2});
    if ~any([2, 4, 8] == conf_num)
        error('CSF_WM must be 2, 4, or 8.');
    else
        phys_id = floor(conf_num / 2);
        if phys_id > 0, selected_keys = [selected_keys, {'white_matter', 'csf'}]; end %#ok<AGROW>
        if phys_id > 1, selected_keys = [selected_keys, {'white_matter_derivative1', 'csf_derivative1'}]; end %#ok<AGROW>
        if phys_id > 2, selected_keys = [selected_keys, {'white_matter_derivative1_power2', 'csf_derivative1_power2', 'white_matter_power2', 'csf_power2'}]; end %#ok<AGROW>
    end
end

% aCompCor
if any(contains(pipeline, 'aCompCor'))
    csf_50_dict = json_run(ismember({json_run.Mask}, 'CSF') & ismember({json_run.Method}, 'aCompCor') & ~contains({json_run.key}, 'dropped'));
    wm_50_dict  = json_run(ismember({json_run.Mask}, 'WM')  & ismember({json_run.Method}, 'aCompCor') & ~contains({json_run.key}, 'dropped'));
    idx = find(contains(pipeline, 'aCompCor'));
    conf_num_str = pipeline{idx(1)};
    conf_num = str2double(strsplit(conf_num_str{1}, '-'){2});
    if ~any([10, 50] == conf_num)
        error('aCompCor must be 10 or 50.');
    else
        if conf_num == 10
            csf = sort(cell2mat(csf_50_dict.keys())); csf_10 = csf(1:5);
            wm  = sort(cell2mat(wm_50_dict.keys()));  wm_10  = wm(1:5);
            selected_keys = [selected_keys, csf_10, wm_10]; %#ok<AGROW>
        else
            csf_50 = cell2mat(csf_50_dict.keys());
            wm_50  = cell2mat(wm_50_dict.keys());
            selected_keys = [selected_keys, csf_50, wm_50]; %#ok<AGROW>
        end
    end
end

% Cosine
if any(contains(pipeline, 'Cosine'))
    cosine_keys = tsv_run.Properties.VariableNames(contains(tsv_run.Properties.VariableNames, 'cosine'));
    selected_keys = [selected_keys, cosine_keys]; %#ok<AGROW>
end

% MotionOutlier
if any(contains(pipeline, 'MotionOutlier'))
    motion_outlier_keys = tsv_run.Properties.VariableNames(find(contains(tsv_run.Properties.VariableNames, {'non_steady_state_outlier', 'motion_outlier'})));
    selected_keys = [selected_keys, motion_outlier_keys]; %#ok<AGROW>
end

% Framewise Displacement (FD)
if any(contains(pipeline, 'FD'))
    fd_values = tsv_run.framewise_displacement;
    if isnan(fd_values(1)), fd_values(1) = 0; end
    tsv_run.framewise_displacement = fd_values;
    selected_keys = [selected_keys, {'framewise_displacement'}]; %#ok<AGROW>
end

% Retrieve selected confounds and fill missing with zeros
confounds_table = tsv_run(:, ismember(tsv_run.Properties.VariableNames, selected_keys));
confounds = fillmissing(confounds_table, 'constant', 0);

