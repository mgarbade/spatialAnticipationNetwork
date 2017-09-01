function convertMatResultsToPNG(inputDir,outputDir,imagesDir,dbMode)
% 


if nargin < 3
    imagesDir = '/home/garbade/datasets/VOC2012/JPEGImages';
end

if nargin < 4
%     dbMode = 1; % Voc12
    dbMode = 3; % CamVid
%     dbMode = 5; % Cityscapes
end

searchString = [inputDir '/*_blob_0.mat'];
img_filelist = getFilesInFolder(searchString);

numFiles = size(img_filelist,1);

names = cell(numFiles,1);
heights_orig = zeros(numFiles,1);
widths_orig = zeros(numFiles,1);
for i = 1:numFiles

    split_points = strfind(img_filelist{i},'_');
    names{i} = img_filelist{i}(1:split_points(end-1)-1);

    % img = imread([imagesDir '/' names{i}(1:11) '.jpg']);
    
    switch(dbMode)
        case 1
            % Voc12
%             img = imread([imagesDir '/' names{i}(1:end-4) '.jpg']);
%             img = imread([imagesDir '/' names{i} '.jpg']);
            img = imread([imagesDir '/' names{i}(5:end) '.jpg']);
        case 3
            % CamVid
            img = imread([imagesDir '/' names{i} '.png']);
        case 4
            % vw_dataset
            img = imread([imagesDir '/' names{i} '.png']);
        case 5
            % Cityscapes
            % img = imread([imagesDir '/' names{i} '.jpg']); %
        otherwise
            error('Please specify dataset. 1: Voc12, 3: CamVid, 5: Cityscapes');
    end

    [heights_orig(i),widths_orig(i),~] = size(img);
end


annFiles = cell(numFiles,1);
for i = 1:numFiles
    if ~mod(i,100)
        disp([num2str(i) ' /' num2str(numFiles)]);
    end
    
    annFiles{i} = [inputDir '/' names{i} '_blob_0.mat'];
    seg = load(annFiles{i});
    seg = seg.data;

    if size(seg,3) > 1
        [~, ind] = max(seg,[],3);
        labelImg = ind - 1; %% TODO: Make labelImg 0-based
    else
        labelImg = seg;
    end
    labelImg = labelImg';
    labelImg = labelImg(1:heights_orig(i),1:widths_orig(i),:);

    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    
    % Save colorized img
    imwrite(uint8(labelImg),[outputDir '/' names{i} '.png'],'png'); 
end