function weight_vector = adjust_contrasts(spmMatPath, contrastWeights)
% ADJUST_CONTRASTS Adjust contrast weights according to the design matrix in SPM.
%
% See old-implementation for original version. This copy is used by
% chess-glm to generate autocontrasts via wildcard matching.

% Load SPM.mat
load(spmMatPath);
regressor_names = SPM.xX.name;

% Generate weight vector based on SPM's design matrix and specified weights
weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names);
end

function weight_vector = generate_weight_vector_from_spm(contrastWeights, regressor_names)
% GENERATE_WEIGHT_VECTOR_FROM_SPM Builds a weight vector with optional wildcards.

weight_vector = zeros(1, length(regressor_names));
fields = fieldnames(contrastWeights);

for i = 1:length(fields)
    field = fields{i};
    if contains(field, '_WILDCARD_')
        pattern = ['Sn\(\d{1,2}\) ' strrep(field, '_WILDCARD_', '.*')];
        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));
        weight_vector(idx) = contrastWeights.(field);
    else
        pattern = ['Sn\(\d{1,2}\) ' field];
        idx = find(~cellfun('isempty', regexp(regressor_names, pattern)));
        if ~isempty(idx)
            weight_vector(idx) = contrastWeights.(field);
        end
    end
end
end

