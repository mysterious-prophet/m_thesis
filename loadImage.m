%% Load Image
% load image from a file

% inputs: image_filename - filename of the image being loaded
% outputs: input_image - loaded input image
%        : spect_bool - boolean showing whether input image is a SPECT
%                       image, used for SPECT normalization

function [input_image, spect_bool] = loadImage(image_filename)
    dot_ind = strfind(image_filename, '.');
    dot_ind = dot_ind(1);
    image_format = image_filename(dot_ind+1:end);
    spect_bool = false;

    % supported formats
    dicom_formats = {'dcm', 'dicom'};
    nifti_formats = {'nii', 'nii.gz'};
    interfile_formats = {'img'};

    if(ismember(image_format, dicom_formats))
        [input_image, ~] = dicomread(image_filename);
        input_image = squeeze(double(input_image));
        spect_bool = true;
   
    elseif(ismember(image_format, nifti_formats))
        metadata = niftiinfo(image_filename);
        input_image = niftiread(metadata);
        
    elseif(ismember(image_format, interfile_formats))
        image_filename = image_filename(1:dot_ind-1);
        
        % read .hdr header
        header_file = fopen([image_filename '.hdr'],'r');
        input_header = fread(header_file);
        fclose(header_file);
        
        % 3D .img image sizes and define image
        m = double(input_header(43) + 256*input_header(44));
        n = double(input_header(45) + 256*input_header(46));
        p = double(input_header(47) + 256*input_header(48));
        X = uint32(zeros(m, n, p));
        
        % read .img 1D vector
        img_file = fopen([image_filename '.img'],'r');
        input_image = fread(img_file);
        fclose(img_file);
        
        % create 3D image by transforming data with step size
        step_size = length(input_image)/m/n/p;
        cur_ind = 1;
        for k = 1:p
            for j = 1:n
                for i = 1:m
                    X(i, j, k) = uint32(input_image(cur_ind));
                    if(step_size >= 2 && step_size <= 4)
                        X(i, j, k) = X(i, j, k) + uint32(256*sum(input_image(cur_ind + step_size - 1:cur_ind+1)));
                    end
                    cur_ind = cur_ind + step_size;
                end
            end
        end
        
        input_image = X;
        spect_bool = true;
    end 

    input_image = double(input_image);
end