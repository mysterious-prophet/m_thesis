%% Normalize SPECT Image
% normalize SPECT image by either dividing it by global maximum or sum of
% all elements

% inputs: input_image - non-normalized input image
%       : norm - normalization type
% outputs: input_image - normalized input image

function input_image = normSpectImage(input_image, norm)
    if(norm == 1)
        max_input = max(max(max(input_image)));
        input_image = input_image / max_input;
    elseif(norm == 2)
        sum_input = sum(sum(sum(input_image)));
        input_image = input_image / sum_input;
    end
end