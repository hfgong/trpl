


% Generate superpixel segmentation for every frames

img_dir = 'images/'

% path & input argument for the external superpixel segmentation binary
%

pixelScale    = 1;
%pixelScale    = 0.5;

if pixelScale >=0.6
  if isunix
    segment_command = './superpixel/segment 0.8 100 20 ';
    %segment_command = './superpixel/segment 0.5 50 10 ';
  else
    segment_command = 'segment 0.8 100 20 ';
    %segment_command = 'segment 0.5 50 20 ';

  end
else
  if isunix
    %segment_command = './superpixel/segment 0.8 100 20 ';
    segment_command = './superpixel/segment 0.5 50 10 ';
  else
    %segment_command = 'segment 0.8 100 20 ';
    segment_command = 'segment 0.5 50 20 ';
  end

end


% img_dir = '../data/seq4/left_rect';

workspace   = 'workspace/superpixel';
if ~exist(workspace, 'dir')
    mkdir(workspace);
end

%filesLeft = dir(fullfile(img_dir, '*.jpg'));
filesLeft = textread([img_dir 'image_list_l.txt'], '%s');

for idx = 1:length(filesLeft)
% for idx = 50
    % Read images
    
    fprintf('Generate Segmentation %d: %s...\n', idx, filesLeft{idx});

    tic;
    [a b c] = fileparts(filesLeft{idx});
    if pixelScale>=0.6
      segFile   = fullfile(workspace, sprintf('%s_seg.txt', b));
    else
      segFile   = fullfile(workspace, sprintf('%s_seg_half.txt', b));
    end
    %if exist(segFile, 'file'), continue, end

    % Read images
    I1 = imresize(imread(fullfile([img_dir 'left_rect/'], filesLeft{idx})), pixelScale);
 
    % Run superpixel segmentation code(external binary)
    imwrite(I1, fullfile(workspace, 'input_tmp.ppm'));
    if pixelScale>=0.6
      segFilePPM = fullfile(workspace, sprintf('%s_seg.ppm', b));
    else
      segFilePPM = fullfile(workspace, sprintf('%s_seg_half.ppm', b));
    end

    
    system([segment_command, fullfile(workspace, 'input_tmp.ppm'), ' ', segFilePPM]);
    
    spImg = imread(segFilePPM);
    
    seg = int32(loadSPimage(spImg));
                        
    save(segFile, '-ascii', 'seg');

    toc;
end
