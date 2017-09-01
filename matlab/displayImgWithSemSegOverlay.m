function displayImgWithSemSegOverlay( ...
    filelist,...
    resSemSegRoot,...
    inputImgRoot,...
    dstDir, ...
    varargin)
    
% Cityscapes
labelsFile = '/home/garbade/datasets/cityscapes/labels.mat';

colMap = load(labelsFile);
colorNames = colMap.colorNames; %% TODO: load labels
colors = colMap.colors;    

hFig = figure('Name','Label Prediction','Position', [760 1060 1160 655]);

if ~exist(dstDir,'dir')
    mkdir(dstDir)
end

numFiles = size(filelist,1);
for i = 1:numFiles
    if ~mod(i,100) 
        disp(i)
    end
    
    labelName = [resSemSegRoot filelist{i}];
    if exist([labelName '.png'],'file')
        labelName = [labelName '.png'];
        labelImg = imread(labelName);
    elseif exist([labelName '_gtFine_labelTrainIds.png'],'file')
        labelName = [labelName '_gtFine_labelTrainIds.png'];
        labelImg = imread(labelName);
    elseif exist([labelName '_blob_0.mat'],'file')
        labelName = [labelName '_blob_0.mat'];
        labelImg = load(labelName);
        labelImg = labelImg.data';
    else
        error(['No label found called ' labelName])
    end
    
    % Cityscapes
    postfix = '';
    imgName = [inputImgRoot filelist{i} postfix];
    
    if exist([imgName '.jpg'],'file')
        imgName = [imgName '.jpg'];
    elseif exist([imgName '.png'],'file')
        imgName = [imgName '.png'];
    else
        error(['No image found called ' imgName])
    end
    
    % img = imread([dataRoot filelist{i} '.jpg']);
    img = imread(imgName);
    displayImgWithOverlay(img,labelImg,colorNames,colors);


    % Save as img
    outName = [dstDir '/' filelist{i}];
    saveas(gca,outName,'jpg')
    export_fig([outName '.jpg']);

end



