function run_subject_glm(subPath, subName, selectedTasks, selectedRuns, ...
                         fmriprepRoot, BIDSRoot, outRoot, ...
                         tempDir, pipeline, niftiSpace, thresholds)
    % RUN_SUBJECT_GLM Run first-level SPM model and autocontrasts for one subject.
    %
    % This function mirrors the logic from old-implementation while:
    %  - Removing figure generation (kept for 91_* scripts)
    %  - Preserving autocontrast setup via wildcard matching
    %
    fprintf('[%s] Starting GLM for %s\n', datestr(now), subName);

    for selected_task_idx = 1:length(selectedTasks)
        %% SUBJECT AND TASK INFO
        selectedTask = selectedTasks(selected_task_idx).name;
        contrasts    = selectedTasks(selected_task_idx).contrasts;
        smoothBool   = selectedTasks(selected_task_idx).smoothBool;

        % Quick presence check for task files
        fullPath = fullfile(subPath, subName, 'func');
        files = dir(fullPath);
        fileNames = {files.name};
        containsTask = any(contains(fileNames, ['task-', selectedTask]));
        if ~containsTask
            warning('Task %s not found for %s in %s. Skipping..', selectedTask, subName, fullPath);
            continue;
        end

        % Output path (BIDS-like): <DERIVATIVES>/spm-glm/<smooth4|unsmoothed>/<sub-xx>/<task>
        outPath = fullfile(outRoot, subName, selectedTask);
        if ~exist(outPath, 'dir'), mkdir(outPath); end
        fprintf('############################### \n# STEP: running %s - %s #\n############################### \n', subName, selectedTask)

        % Paths to subject-level fMRIPrep and BIDS
        funcPathSub = fullfile(fmriprepRoot, subName, 'func');
        bidsPathSub = fullfile(BIDSRoot, subName, 'func');

        %% Locate runs/events/confounds
        if ismember('*', selectedRuns)
            eventsTsvFiles = dir(fullfile(bidsPathSub, strcat(subName,'_task-',selectedTask,'_run-*_events.tsv')));
            json_confounds_files = dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask,'_run-*_desc-confounds_timeseries.json')));
            tsv_confounds_files  = dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask,'_run-*_desc-confounds_timeseries.tsv')));
        else
            selected_runs = selectedRuns; %#ok<NASGU> (keep signature compatibility)
            eventsTsvFiles = arrayfun(@(x) dir(fullfile(bidsPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_events.tsv'))), selected_runs, 'UniformOutput', true);
            json_confounds_files = arrayfun(@(x) dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_desc-confounds_timeseries.json'))), selected_runs, 'UniformOutput', true);
            tsv_confounds_files  = arrayfun(@(x) dir(fullfile(funcPathSub, strcat(subName,'_task-',selectedTask, '_run-', sprintf('%01d', x), '_desc-confounds_timeseries.tsv'))), selected_runs, 'UniformOutput', true);
        end

        % Sort by name for stable pairing
        eventsTsvFiles       = table2struct(sortrows(struct2table(eventsTsvFiles), 'name'));
        json_confounds_files = table2struct(sortrows(struct2table(json_confounds_files), 'name'));
        tsv_confounds_files  = table2struct(sortrows(struct2table(tsv_confounds_files), 'name'));

        assert(numel(eventsTsvFiles) == numel(json_confounds_files) && numel(json_confounds_files) == numel(tsv_confounds_files), ...
            'Mismatch in number of TSV events, TSV confounds, and JSON confounds files in %s', funcPathSub)

        %% SPM MODEL (run-independent params)
        matlabbatch{1}.spm.stats.fmri_spec.dir = {outPath};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

        % Model estimation step placeholder (filled after run loop)
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outPath, 'SPM.mat')};

        %% Per-run specification
        for runIdx = 1:numel(eventsTsvFiles)
            % Pair TSV/JSON for the same run
            selectedRun = findRunSubstring(eventsTsvFiles(runIdx).name); % e.g., 'run-2'

            jsonRows = json_confounds_files; %#ok<NASGU>
            tsvRows  = tsv_confounds_files;  %#ok<NASGU>
            jsonRow  = eventsTsvFiles(runIdx); % indices aligned after sorting
            confoundsRow = tsv_confounds_files(runIdx);

            % Build confounds table from fMRIPrep
            confounds_array = fMRIprepConfounds2SPM(fullfile(confoundsRow.folder, strrep(confoundsRow.name,'.tsv','.json')),...
                fullfile(confoundsRow.folder, confoundsRow.name), pipeline);

            % Locate NIfTI for this run (prefer .nii, fallback to .nii.gz and gunzip)
            spaceString = getSpaceString(niftiSpace);
            filePattern = strcat(subName,'_task-', selectedTask, '_', selectedRun,'_space-',spaceString,'_desc-preproc_bold');
            niiFileStruct = dir(fullfile(funcPathSub, strcat(filePattern, '.nii')));

            if numel(niiFileStruct) > 1
                error('Multiple NIFTI files found for %s.', selectedRun)
            elseif isempty(niiFileStruct)
                niiGzFileStruct = dir(fullfile(funcPathSub, strcat(filePattern, '.nii.gz')));
                if isempty(niiGzFileStruct)
                    warning('No NIFTI file found for %s. Skipping run.', selectedRun)
                    continue
                elseif numel(niiGzFileStruct) > 1
                    error('Multiple NIFTI.GZ files found for %s.', selectedRun)
                else
                    niiGzFileString = fullfile(niiGzFileStruct.folder, niiGzFileStruct.name);
                    gunzippedNii = gunzipNiftiFile(niiGzFileString, fullfile(tempDir, 'gunzipped', subName));
                    niiFileStruct = dir(gunzippedNii{1});
                end
            end

            niiFileString = fullfile(niiFileStruct.folder, niiFileStruct.name);

            % Optional smoothing
            if smoothBool
                % Smooth functional image at first level using 4 mm FWHM by default
                niiFileString = smoothNiftiFile(niiFileString, fullfile(tempDir, 'smoothed', subName), [4 4 4]);
            else
                fprintf('SMOOTH: smoothBool=false; skipping smoothing for this task.\n')
            end

            % Events TSV
            eventsTable = readtable(fullfile(eventsTsvFiles(runIdx).folder, eventsTsvFiles(runIdx).name), 'FileType','text');
            events_struct = bids_events_to_conditions(eventsTable);

            % Assign run specifics
            matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).scans = spm_select('expand', {niiFileString});
            % HPF effectively disabled by setting above empirical duration
            matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).hpf = (matlabbatch{1}.spm.stats.fmri_spec.timing.RT * size(matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).scans, 1)) + 100;

            for cond_id=1:length(events_struct.names)
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).name    = events_struct.names{cond_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).onset   = events_struct.onsets{cond_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).cond(cond_id).duration= events_struct.durations{cond_id};
            end

            % Confounds as regressors
            for reg_id=1:size(confounds_array,2)
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).regress(reg_id).name = confounds_array.Properties.VariableNames{reg_id};
                matlabbatch{1}.spm.stats.fmri_spec.sess(runIdx).regress(reg_id).val  = confounds_array{:, reg_id};
            end
        end

        %% RUN SPECIFICATION AND ESTIMATION
        spm('defaults','fmri'); spm_jobman('initcfg');
        fprintf('GLM: Running GLM for: %s - TASK: %s\n', subName, selectedTask)
        spm_jobman('run', matlabbatch(1:2));
        fprintf('GLM: DONE!\n')

        %% CONTRASTS (autocontrast)
        spmMatPath = fullfile(outPath, 'SPM.mat');
        if ~exist(spmMatPath, 'file')
            error('SPM.mat missing in %s', outPath);
        end

        matlabbatch{3}.spm.stats.con.spmmat(1) = {spmMatPath};
        for k = 1:length(contrasts)
            weights = adjust_contrasts(spmMatPath, selectedTasks(selected_task_idx).weights(k));
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.weights = weights;
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.name    = contrasts{k};
            matlabbatch{3}.spm.stats.con.consess{k}.tcon.sessrep = 'none';
        end
        fprintf('GLM: Setting contrasts..\n');
        spm_jobman('run', matlabbatch(3));
        fprintf('GLM: Contrasts DONE!\n');

        %% Save a copy of this function into the subject output for provenance
        try
            FileNameAndLocation=[mfilename('fullpath')];
            script_outpath=fullfile(outPath,'spmGLMautoContrast.m');
            currentfile=strcat(FileNameAndLocation, '.m');
            copyfile(currentfile,script_outpath);
        catch ME
            warning('Could not copy script for provenance: %s', ME.message);
        end
    end
end

%% ============================= Helpers ===================================
function runSubstring = findRunSubstring(inputStr)
pattern = 'run-\d{1,2}';
matches = regexp(inputStr, pattern, 'match');
if ~isempty(matches), runSubstring = matches{1}; else, runSubstring = ''; end
end

function spaceString = getSpaceString(niftiSpace)
    if strcmpi(niftiSpace, 'T1w')
        spaceString = 'T1w';
    elseif strcmpi(niftiSpace, 'MNI')
        spaceString = 'MNI152NLin2009cAsym*';
    else
        error('Invalid niftiSpace value');
    end
end

function events = bids_events_to_conditions(tbl)
    % Convert BIDS events to SPM condition struct (names, onsets, durations)
    if ~all(ismember({'onset','duration','trial_type'}, tbl.Properties.VariableNames))
        error('Events TSV must contain onset, duration, trial_type columns');
    end
    utypes = unique(tbl.trial_type, 'stable');
    events.names = cellfun(@char, utypes, 'UniformOutput', false);
    events.onsets = cell(numel(utypes),1);
    events.durations = cell(numel(utypes),1);
    for i=1:numel(utypes)
        mask = strcmp(tbl.trial_type, utypes{i});
        events.onsets{i} = tbl.onset(mask);
        events.durations{i} = tbl.duration(mask);
    end
end
