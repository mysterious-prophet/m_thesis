%% Get Array String
% for certain variable we have arrays of inputs, e.g. filter_name = [noFilt
% fftLP] and thus we must make these into one string

% inputs: array_name - array of parameters
% outputs: array_string - array converted into one string

function array_string = getArrayString(array_name)
    array_string = '';
    for i = 1:size(array_name, 2)
        array_string = strcat(array_string, string(array_name(i)), '_');
    end
    array_string = extractBetween(array_string, 1, strlength(array_string) - 1);
end