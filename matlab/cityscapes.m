%% Create a dataset

dataDir = '/home/garbade/models/05_Cityscapes/01_DL_v2_Johann_CityscapesFromScratch/city/list/filelists';

imgs = getNamesFromAsciiFile('leftImg8bit_train.txt');
labels = getNamesFromAsciiFile('gt_fine_train.txt');

imgRoot = '/leftImg8bit_trainvaltest/leftImg8bit/';
gtRoot = '/gtFine_trainvaltest/gtFine/';

numImgs = size(imgs,1);
for i = 1:numImgs
    [res,msg]=system(['echo "' imgRoot imgs{i}(2:end) ' ' gtRoot labels{i}(2:end) '" >> train_aug_files.txt']);
    if res
        disp(msg)
    end
end




%% Resize images with nearest neighbour interpolation
filelist = getNamesFromAsciiFile('filelist_images.txt');

dataDir = 'images/';
% postFix = '_gtFine_labelTrainIds.png';
postFixImg = '';


numImgs = size(filelist,1);
for i = 1:numImgs
    im = imread([dataDir filelist{i} postFixImg]);
    im_small = imresize(im, 0.5, 'nearest');
    imwrite(uint8(im_small), ['images_50/' filelist{i} postFixImg],'png');
end

for i = 1:numImgs
    im = imread([dataDir filelist{i} postFixImg]);
    im_small = imresize(im, 0.25, 'nearest');
    imwrite(uint8(im_small), ['images_25/' filelist{i} postFixImg],'png');
end


filelist = getNamesFromAsciiFile('filelist_images.txt');
numImgs = size(filelist,1);
maxLabel = 0;
for i = 1:numImgs
    im = imread(['images_50/' filelist{i} postFixImg]);
    im_red = im;
    im_red(im == 255) = [];
    curMaxLabel = max(im_red(:));
    if curMaxLabel > maxLabel;
        maxLabel = curMaxLabel;
    end
%     if strcmp(class(im),'uint8')
%         disp([num2str(i) ': ' filelist{i} ' is uint8'])
%     elseif strcmp(class(im),'uint16')
%         disp([num2str(i) ': ' filelist{i} ' is uint16'])
%     end
end


%% Split validation images in 2 pieces
dataDir = '/home/garbade/models/05_Cityscapes/01_DL_v2_Johann_CityscapesFromScratch/city/list/filelists';
cd(dataDir)
imgs = getNamesFromAsciiFile('leftImg8bit_val.txt');
labels = getNamesFromAsciiFile('gt_fine_val.txt');
numImgs = size(imgs,1);
for i = 1:numImgs
    if ~mod(i,50) 
        disp(i)
    end
    [path, name, ext] = fileparts(imgs{i});
    name_id = name(1:end-12);
    
    % system(['echo ' name_id ' >> ' 'val_id.txt']);
    
    gtSeg = imread(['gtFine/images_50/' name_id '_gtFine_labelTrainIds.png']);
    img = imread(['leftImg8bit/images_50/' name_id  '_leftImg8bit.jpg']);
    
    if ~ (sum([512 1024] == size(gtSeg)) == 2) 
        error('Size of input image is not [512 1024] as it''s supposed to be.')
    end
    
    offset = 0;
    gtSeg_left = gtSeg(1:512,1:512 + offset);
    gtSeg_right = gtSeg(1:512,513 - offset:1024);

    img_left = img(1:512,1:512 + offset,:);
    img_right = img(1:512,513 - offset:1024,:);
    
    gtSeg_left_name = ['labels_50_val_split/' name_id '_left.png']; 
    gtSeg_right_name = ['labels_50_val_split/' name_id '_right.png'];
    img_left_name = ['images_50_val_split/' name_id  '_left.jpg']; 
    img_right_name = ['images_50_val_split/' name_id  '_right.jpg'];
    imwrite(uint8(gtSeg_left), gtSeg_left_name);
    imwrite(uint8(gtSeg_right), gtSeg_right_name);
    imwrite(uint8(img_left), img_left_name);
    imwrite(uint8(img_right), img_right_name);
    
    system(['echo ' img_left_name ' ' gtSeg_left_name ' >> ' 'val_split_left.txt']);
    system(['echo ' img_right_name ' ' gtSeg_right_name ' >> ' 'val_split_right.txt']);
    
    
end

        
        
mkdir gtFine/images_50_split_offset/
mkdir leftImg8bit/images_50_split_offset/     
        
%% Evaluate 
addpath('/home/garbade/Dropbox/MatlabSrc/Utilities/cityscapes')
addpath('/home/garbade/Dropbox/MatlabSrc/Utilities/voc12/')

inputDir = '/home/garbade/models/05_Cityscapes/03_DL_v2_ResNet/city/features/crf/';
outputDir = 'res_02_split_crf_577';
imagesDir = '/home/garbade/datasets/cityscapes/images_50_val_split_offset/';
convertMatResultsToPNG(inputDir,outputDir,imagesDir)
        
        
inputDir = '/home/garbade/models/05_Cityscapes/02_DL_v2_Johann_CityscapesFromScratch_CUBE/city/features_crf/';
outputDir = 'res_02_split_crf';
imagesDir = '/home/garbade/datasets/cityscapes/leftImg8bit/images_50_split/';
convertMatResultsToPNG(inputDir,outputDir,imagesDir)



%% Create MASK with drecreasing label weights
filelistDir = '~/datasets/cityscapes/';
allFiles = '/home/garbade/datasets/cityscapes/filelists/allFilesFine_id.txt';
filelist = getNamesFromAsciiFile(allFiles); % MODIFY
bgr = [104 116 122];
rgb = bgr([3 2 1]);
% yc = 504/2;
% xc = 672/2;

yc = 512/2;
xc = 1024/2;


offset = 160;
ymin = yc - offset;
ymax = yc + offset;
xmin = xc - 2*offset;
xmax = xc + 2*offset;
imgDir = '~/datasets/cityscapes/images_50/';
gtSegDir = '~/datasets/cityscapes/labels_50/';
dstImgDir = '~/datasets/cityscapes/images_50_MeanBar/';
dstGtDir = '~/datasets/cityscapes/labels_50_MatWithMask_uint8/';

postFixImg = '_leftImg8bit.jpg';
postFixLab = '_gtFine_labelTrainIds.png';
numImgs = size(filelist,1);
for i = 1:numImgs
    if ~mod(i,50)
        disp([num2str(i) ' /' num2str(numImgs)]);
    end
%     im = imread([imgDir filelist{i} postFixImg]);
    gtLabel = imread([gtSegDir filelist{i} postFixLab]);
    
    mask = ones(size(gtLabel));
    mask = squeeze(mask(:,:,1));
    mask(ymin:ymax,xmin:xmax,1) = 0;
    
%     img_tmp = im;
%     img_1 = squeeze(img_tmp(:,:,1));
%     img_2 = squeeze(img_tmp(:,:,2));
%     img_3 = squeeze(img_tmp(:,:,3));
% 
%     img_1(logical(mask)) = rgb(1);
%     img_2(logical(mask)) = rgb(2);
%     img_3(logical(mask)) = rgb(3);
%     img_tmp = cat(3, img_1, img_2, img_3);
%     imwrite(uint8(img_tmp),[dstImgDir filelist{i} '.png'],'png');
    
    newLabel = gtLabel;
    newLabel(~logical(mask)) = 11 ;
    
%     % Create decreasing mask
%     dst = cv.distanceTransform(uint8(mask));
%     max_dist_val = max(dst(:));
%     dst_new = -1/max_dist_val .* dst + 1;
%     dst_new(dst_new < 0) = 0;
%     
    % single
%     new_mask = cat(3,single(gtLabel),single(newLabel),dst_new);

    % uint8 
    new_mask = cat(3,uint8(gtLabel),uint8(newLabel));
    
    allMasks = new_mask;
    for j = 0 : 11
       gt_ch = (gtLabel == j);
       allMasks = cat(3,allMasks,uint8(gt_ch));
    end
    
    % single
%     data = single(allMasks);
    % uint8 
    data = uint8(allMasks);

    save([dstGtDir filelist{i} '.mat'],'data');
    
    
%     FileAndLabel = ['/images_70_MeanBar/' filelist{i} '.png' ' ' '/labels_70_MatWithMask/' filelist{i} '.mat'];
%     system(['echo ' FileAndLabel ' >> test.txt'])
end


Exp = '10_C3_fullMask_Test';
inputDir = ['/home/garbade/models/05_Cityscapes/' Exp '/city/features/no_crf_577/'];
outputDir = 'res_01_split';
mkdir(outputDir)
imagesDir = '/home/garbade/datasets/cityscapes/images_50_val_split_offset/';
convertMatResultsToPNG(inputDir,outputDir,imagesDir)



% Display Results as overlay
% Exp = '09_C5_Test';
Exp = '09_Combine_C1_C5';
% inputImgRoot = '/home/garbade/datasets/cityscapes/images_50/'; % input img
inputImgRoot = '/home/garbade/datasets/cityscapes/images_50_MeanBar/';
ExpDir = ['/home/garbade/models/05_Cityscapes/' Exp '/'];
% resSemSegRoot = [ ExpDir '/res_01_combined/']; % png encoding the pred classes from
% resSemSegRoot = [ ExpDir '/combine_C1_C5_Gt_dlt_0/']; 
resSemSegRoot = '/home/garbade/datasets/cityscapes/labels_50/'; 
dstDir = [ ExpDir '/gt_overlay2/'];
mkdir(dstDir)
mkdir(resSemSegRoot)
filelist = getNamesFromAsciiFile('~/datasets/cityscapes/val_id.txt');

displayImgWithSemSegOverlay( ...
                            filelist,...
                            resSemSegRoot,...
                            inputImgRoot,...
                            dstDir,...
                            'dbMode',5)

                        
                        
% Combine results from left and right images
inputImgRoot = '/home/garbade/datasets/cityscapes/images_50/'; % input img
Exp = '10_C3_fullMask_Test';
ExpDir = ['/home/garbade/models/05_Cityscapes/' Exp '/'];
resSemSegRoot = [ ExpDir '/city/res_01_split/']; % png encoding the pred classes from
dstDir = [ ExpDir '/city/res_01_combined/'];
mkdir(dstDir)
filelist = getNamesFromAsciiFile([ExpDir '/city/list/val_id.txt']);
numImgs = size(filelist,1);
for i = 1:numImgs
    if ~mod(i,50)
        disp([num2str(i) ' /' num2str(numImgs)]);
    end
    imname = filelist{i};
    % Crop imgs to 512
    % left    
    leftfile = [resSemSegRoot '/' imname '_left.png'];
    [leftim,map] = imread(leftfile);
    leftim = leftim(:,1:512);
    % right
    rightfile = [resSemSegRoot '/' imname '_right.png'];
    [rightim,map] = imread(rightfile);
    rightim = rightim(:,65:end-1);
    resim = [leftim, rightim];
    imwrite(uint8(resim),[dstDir '/' imname '.png'],'png'); 
end


% Evaluate Results
Exp = '03_DL_v2_ResNet';
cd(['~/models/05_Cityscapes/' Exp '/city/'])
% sz50
mask_size = 50;
gtDir = '~/datasets/cityscapes/labels_50/';
resDir = 'images_val_pred_ind_masked_sz50';
% sz100
start = tic;
mask_size = 100;
gtDir = '~/datasets/cityscapes/labels/';
% resDir = 'val_val_sz100_phase2_predNet_641x1281_colors_23_ind/';
% resDir = 'gt_ind_masked_sz100/';
% resDir = 'val_val_sz100_rgb_input_ind/';
% resDir = 'val_val_phase2_642x1282_colors_23_ind/'
% resDir = 'val_val_phase3_833x1665_colors_23_ind/'
resDir = 'images_val_pred_ind_masked_sz100/';
pixel_acc = zeros(3,1);
class_acc = zeros(3,1);
IoU = zeros(3,1);
conf = cell(3,1);
[pixel_acc(1),class_acc(1),IoU(1),conf{1}] = city_evalSeg(resDir,gtDir,mask_size,'MaskMode',1);
[pixel_acc(2),class_acc(2),IoU(2),conf{2}] = city_evalSeg(resDir,gtDir,mask_size,'MaskMode',2);
[pixel_acc(3),class_acc(3),IoU(3),conf{3}] = city_evalSeg(resDir,gtDir,mask_size,'MaskMode',3);
IoU
displayElapsedTime(start)


city_evalSeg(resDir,gtDir,mask_size,'MaskMode',3,'n_classes',20);


%% Deloc Acc
resDir = 'val_prob_masked_sz100/';
gtDir = '~/datasets/cityscapes/labels/';
mask_size = 100;
k = 3;
stride = 1;
[accAll] = city_evalSeg_delocLoss(resDir,...
                                gtDir,...
                                mask_size,...
                                k,...
                                stride);



%% Split mean images in two halves
filelist = getNamesFromAsciiFile('val_id.txt');
xc = 1024/2;
imgDir = '~/datasets/cityscapes/images_50_MeanBar/';
dstImgDir = '~/datasets/cityscapes/images_50_MeanBars_val_split_offset/';
postFixImg = '.png';
numImgs = size(filelist,1);
for i = 1:numImgs
    if ~mod(i,50)
        disp([num2str(i) ' /' num2str(numImgs)]);
    end
    img = imread([imgDir filelist{i} postFixImg]);

    % Split in half with offset
    offset = 65;
    img_left = img(1:512,1:512 + offset,:);
    img_right = img(1:512,513 - offset:1024,:);
    
    img_left_name = [dstImgDir filelist{i}  '_left.png']; 
    img_right_name = [dstImgDir filelist{i}  '_right.png'];
    imwrite(uint8(img_left), img_left_name);
    imwrite(uint8(img_right), img_right_name);
end



%% Disp imgs combined

% Mask for Cityscapes
yc = 512/2;
xc = 1024/2;
offset = 160;
ymin = yc - offset;
ymax = yc + offset;
xmin = xc - 2*offset;
xmax = xc + 2*offset;
mask = ones(512,1024);
mask(ymin:ymax,xmin:xmax,1) = 0;

labelsFile = '/home/garbade/datasets/cityscapes/labels.mat';
colMap = load(labelsFile);
colorNames = colMap.colorNames; %% TODO: load labels
colors = colMap.colors;  
imgDir = '~/datasets/cityscapes/images_50_MeanBar/';
predSegDirInside = ['C1_res_02/'];
predSegDirOutside = ['C3_fullMask/'];
% dstImgDir = 'res_01_combine_C1_C2';
dstGtDir = 'combine_C1_C3_Gt_dlt_0/';
dstGtDir_predLab = 'combine_C1_C3_Gt_dlt_0_predLab/';
mkdir(dstGtDir)
mkdir(dstGtDir_predLab)


postFix = '.png';
numImgs = size(filelist,1);
for i = 1:numImgs
    if ~mod(i,50)
        disp([num2str(i) ' /' num2str(numImgs)]);
    end
    im = imread([imgDir filelist{i} postFix]);
    predLabelInside = double(imread([predSegDirInside filelist{i} postFix]));
    predLabelOutside = double(imread([predSegDirOutside filelist{i} postFix]));
    
    mask = ones(size(im));
    mask = squeeze(mask(:,:,1));
    
    dlt = 0;
    mask(ymin - dlt:ymax+dlt,xmin - dlt:xmax + dlt,1) = 0;
    
    predCombined =  (1 - mask) .* predLabelInside + mask .* predLabelOutside;
    predCombined = uint8(predCombined);
    imwrite(predCombined,[dstGtDir filelist{i} '.png'],'png');
    
%     imagesc(predCombined);
%     rectangle('Position',[xmin ymin (xmax-xmin) (ymax-ymin)]);
%     pause(1)
    overlay_final = ind2rgb(predCombined,colors);
    overlay_final = uint8(overlay_final);
    imwrite(overlay_final,[dstGtDir_predLab filelist{i} '_predLab.png'],'png');
    
end




%% Switch from 255 to 19 -> [720 960] -> [504 672]
numImgs = size(filelist,1);
for i = 1:numImgs
    gtLabel = imread(['labels/' filelist{i} '.png']);
    gtLabel(gtLabel == 255) = 19;
    
    imwrite(uint8(gtLabel),['labels_ic19/' filelist{i} '.png'],'png');
end
    


%% Create train imgs with black boxes
filelist = getFilesInFolder('images/*.png',true);
imgDir = 'images/';
% filelist = getNamesFromAsciiFile('images/train_aug.txt');
numImgs = size(filelist,1);
% dstDir = 'labels_ic19/';
labels = {'03' '05'};
ratios = [0.3 0.5];
for z = 1:2
    dstDir = ['masks_50/02_random/' labels{z} '/'];
    mkdir(dstDir)
    allRects = zeros(numImgs,4);
    for i = 1:numImgs
        if ~mod(i,100) 
            disp(i)
        end
        img_name = [filelist{i} '.png'];
    %     gtSeg = imread( [imgDir img_name]);

        mask = ones(size(gtSeg));
        [rect] = getRandomRectangle(gtSeg,ratios(z),ratios(z));
        allRects(i,:) = rect;

        % Create a mean val box
        mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),1) = 0;
        imwrite(uint8(mask),[dstDir filelist{i} '.png'],'png');

    %     if (~mod(i,10))
    %       imagesc(mask);
    %       drawnow;
    %       pause(0.300)
    %     end
    end
end


%%% Create center crop
step = 0.1;
for i = 1:6
    minRatio = step*i;
    new_mask = ones(size(mask));
    height = size(new_mask,1);
    width = size(new_mask,2);

    minWidth = floor(width * minRatio);
    maxWidth = width - minWidth;

%     new_mask(:,[1:minWidth, maxWidth:end])=0;
%     new_mask(:,[maxWidth:end])=0; % right side
    new_mask(:,[1:minWidth])=0; % left side
    imagesc(new_mask);
    colorbar;
    pause(1)
    imwrite(uint8(new_mask),['mask_' num2str(i) '0.png'],'png');
end

%%%% Split images
filelist = getFilesInFolder('labels_50_train_val_nc20_128x64_ind_sorted/*.png',true);
numImgs = size(filelist,1);
srcDir = 'labels_50_train_val_nc20_128x64_ind_sorted/';
dstDirLeft = 'labels_50_nc20_82x41_ind/';
% dstDirLeft = 'labels_50_train_val_nc20_128x64_ind_sorted_left/';
% dstDirRight = 'labels_50_train_val_nc20_128x64_ind_sorted_right/';
for i = 1:numImgs
    if ~mod(i,100) 
        disp(i)
    end
    im = imread([srcDir filelist{i} '.png'] );
%     imLeft = im(:,1:64,:);
%     imRight = im(:,65:128,:);

%     waitforbuttonpress
%     pause(1)

    im_small = imresize(im, 1/1.56097561, 'nearest');
    imwrite(uint8(im_small), [dstDirLeft filelist{i} '.png'],'png');
%     imwrite(uint8(imRight), [dstDirRight filelist{i} '.png'],'png');
end


filelist = getNamesFromAsciiFile('filelists/train_id.txt');
numImgs = size(filelist,1);
srcDir = 'images/';
dstDir = 'images_160x320_bicubic/';
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );

    im_small = imresize(im, 160/1024, 'nearest');
    %im_small = imresize(im, 160/1024, 'bicubic');
    imwrite(uint8(im_small), [dstDir filelist{i} '.png'],'png');
end


% % colour map cityscapes
% label_colours = [(128,64,128]
%                 % 0='road'
%                 ,(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153)
%                 % 1='sidewalk', 2='building', 3='wall', 4='fence', 5='pole'
%                 ,(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180)
%                 % 6='traffic light', 7='traffic sign', 8='vegetation', 9='terrain', 10='sky'
%                 ,(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100)
%                 % 11='person', 12='rider', 13='car', 14='truck', 15='bus'
%                 ,(0,80,100),(0,0,230),(119,11,32),(0,0,0)]
%                 % 16='train', 17='motorcycle', 18='bicycle', 19='background'


%% New masks
size_y = 160;
size_x = 320;
offset = 160 *0.2;
ymin = offset;
ymax = size_y - offset;
xmin = 2*offset;
xmax = size_x - 2*offset;
new_mask = zeros(size(mask_small));
new_mask(ymin:ymax,xmin:xmax,1) = 1;
imagesc(new_mask)
imwrite(uint8(new_mask),'mask_160x320_20pc.png')


labelsFile = '/home/garbade/datasets/cityscapes/labels.mat';
colMap = load(labelsFile);
colorNames = colMap.colorNames; %% TODO: load labels
colors = colMap.colors;
overlay_final = ind2rgb(predCombined,colors);
overlay_final = uint8(overlay_final);
imwrite(overlay_final,[dstGtDir_predLab filelist{i} '_predLab.png'],'png');
filelist = getNamesFromAsciiFile('filelists/train_id.txt');
numImgs = size(filelist,1);
srcDir = 'labels_ic19/';
dstDir = 'labels_160x320_ic19_rgb/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_small = imresize(im, 160/1024, 'nearest');
    overlay_final = ind2rgb(im_small,colors);
    overlay_final = uint8(overlay_final);
    imwrite(uint8(overlay_final), [dstDir filelist{i} '.png'],'png');
end



%% Deloc Acc

pred = imread('images_val_pred_ind_masked_sz100/frankfurt_000000_000294.png');
gt = imread('~/datasets/cityscapes/labels_ic19/frankfurt_000000_000294.png');
pred_small = imresize(pred, 10/1024,'nearest') ;
gt_small = imresize(gt, 10/1024,'nearest') ;
pred_one_hot = zeros(size(pred_small));
gt_one_hot = zeros(size(gt_small));
pred_one_hot_single = single(pred_one_hot);
for i = 1:19
    pred_one_hot(:,:,i) = ind2sub(pred_small,pred_small==i);
    gt_one_hot(:,:,i) = ind2sub(gt_small,gt_small==i);
    imagesc(gt_one_hot(:,:,i));colorbar;pause(1)
end
gt_one_hot_single = single(gt_one_hot);
pred_one_hot_single = single(pred_one_hot);
pred_pooled = vl_nnpool(pred_one_hot_single,[3,3],'Pad',[1,1,1,1]);
pred_pooled_5 = vl_nnpool(pred_one_hot_single,[5,5],'Pad',[2,2,2,2]);
gt_pooled = vl_nnpool(gt_one_hot_single,[3,3],'Pad',[1,1,1,1]);
gt_pooled_v5 = vl_nnpool(gt_one_hot_single,[5,5],'Pad',[2,2,2,2]);


for i = 1:19
    figure(1);imagesc(pred_one_hot_single(:,:,i));colorbar;
    figure(2);imagesc(pred_pooled(:,:,i));colorbar;
    pause(2)
end


% Loop over pred labels and gt labels
% Load data
% Convert to one_hot encoding
% Perform pooling operation
% Compare channelwise pred and GT
% Compute 

dataDir = '~/datasets/cityscapes/';
mask_file = '~/datasets/cityscapes/masks/01_center_visible/mask_sz100_1024x2048.png'; 
mask = imread(mask_file);

gtDir = [dataDir 'labels_ic19/'];
srcDir = 'images_val_pred_ind_masked_sz100/';
filelist = getFilesInFolder([srcDir '*.png'],true);
numImgs = size(filelist,1);
n_classes = 19;
k = 1;
mask_outside_rep = repmat(mask_outside,1,1,n_classes);
acc_all = zeros(numImgs,1);
for i = 1:numImgs
    disp(i)
    pred = single(imread([srcDir filelist{i} '.png']));
    gt = single(imread([gtDir filelist{i} '.png']));
    pred_one_hot = zeros(size(pred));
    gt_one_hot = zeros(size(gt));
    for j = 1:n_classes
        pred_one_hot(:,:,j) = ind2sub(pred,pred==j);
        gt_one_hot(:,:,j) = ind2sub(gt,gt==j);
        %imagesc(gt_one_hot(:,:,j));colorbar;pause(1)
    end    
    p = floor(k/2);
    pred_pooled = vl_nnpool(single(pred_one_hot),[k,k],...
                                            'Pad',[p,p,p,p],...
                                            'Stride',1);
    gt_pooled = vl_nnpool(single(gt_one_hot),[k,k],...
                                            'Pad',[p,p,p,p],...
                                            'Stride',1);
    corr = pred_pooled == gt_pooled;
    corr_masked = corr(mask_outside_rep);
    corr_masked_numel = numel(corr_masked);
    corr_masked_sum = sum(corr_masked(:));
    acc_all(i) = corr_masked_sum / corr_masked_numel;
end

system(['echo "Acc_Deloc_Mask_outside_sz100_k1: ' num2str(mean(acc_all)) '" >> res.txt']);


%% Downsample Segmentation masks
filelist = getFilesInFolder('circle_labels_512x1024/train/*.png',true);
numImgs = size(filelist,1);
srcDir = 'circle_labels_512x1024/train/';
dstDir = 'circle_labels_160x320/train/';
mkdir(dstDir)
for i = 1:numImgs
im = imread([srcDir filelist{i} '.png'] );
%     im_small = imresize(im, 160/512, 'bilinear');
    im_small = imresize(im, 160/512, 'nearest');
%     overlay_final = ind2rgb(im_small,colors);
%     overlay_final = uint8(overlay_final);
    imwrite(uint8(im_small), [dstDir filelist{i} '.png'],'png');
end

% Crop validation images
mask = imread('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png');
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
srcDir = 'val_642x1282_sz100/';
dstDir = 'val_642x1282_sz100_padded/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_padded = repmat(mask,1,1,3);
    im_padded(191:832,383:1664,:) = im;
    imwrite(uint8(im_padded), [dstDir filelist{i} '.png'],'png');
end

% Crop images, colored_labels and labels to [642 x 1282]
numImgs = size(filelist,1);
srcDir = 'images/';
dstDir = 'images_val_642x1282/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_cropped = im(191:832,383:1664,:);
    imwrite(uint8(im_cropped), [dstDir filelist{i} '.png'],'png');
end
% labels 
numImgs = size(filelist,1);
srcDir = 'labels_ic19/';
dstDir = 'labels_ic19_val_642x1282/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_cropped = im(191:832,383:1664,:);
    imwrite(uint8(im_cropped), [dstDir filelist{i} '.png'],'png');
end
% colored_labels 
numImgs = size(filelist,1);
srcDir = 'labels_color/';
dstDir = 'labels_color_val_642x1282/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_cropped = im(191:832,383:1664,:);
    imwrite(uint8(im_cropped), [dstDir filelist{i} '.png'],'png');
end

% Prepare images for phase 2
mask = imread('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png');
srcDir = 'val_642x1282/';
dstDir = 'val_642x1282_padded/';
mkdir(dstDir)
for i = 1:numImgs
	im = imread([srcDir filelist{i} '.png'] );
	%im_padded = repmat(mask,1,1,3);
	im_padded = zeros(1024,2048,3);
	im_padded(191:832,383:1664,:)=im;
	imwrite(uint8(im_padded), [dstDir filelist{i} '.png'],'png');
end

%% New Mask
new_mask = zeros(size(mask));
new_mask(96:928,192:1856)=1;
new_mask_filename = '~/datasets/cityscapes/masks/01_center_visible/mask_sz100_phase3_1024x2048.png';
imwrite(uint8(new_name), new_mask_filename,'png');

%% Prepare for phase 3
cd('/media/data1/models_tf/05_Cityscapes/35_3_colors_predNet_Msk1_sz0-5_1-5')
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
srcDir = 'val_phase2_642x1282_colors_23/';
dstDir = 'val_phase2_642x1282_colors_23_crop_for_phase3/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    %im_cropped = im(191:832,383:1664,:);
    im_cropped = im(96:928,192:1856,:);
    im_padded = zeros(1024,2048,3);
    im_padded(96:928,192:1856,:) = im_cropped;
    imwrite(uint8(im_padded), [dstDir filelist{i} '.png'],'png');
end


%% Display image with phase1 and phase 2 border
gtcol = imread('images_val_pred_masked_sz100/frankfurt_000000_000294.png');
figure(2);imshow(im)
rectangle('Position',[192,96,1665,833],'LineWidth',2,'EdgeColor','w')
rectangle('Position',[383,191,1282,642],'LineWidth',2,'EdgeColor','w')

yc = 1024/2;
xc = 2048/2;
offset = 320;
ymin = yc - offset;
ymax = yc + offset;
xmin = xc - 2*offset;
xmax = xc + 2*offset;
mask = ones(1024,2048);
mask(ymin:ymax,xmin:xmax,1) = 0;

%% padding for all-at-once prediction
cd('/media/data1/models_tf/05_Cityscapes/22_msc_fullSizeInput_nc19')
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
srcDir = 'val_semseg_642x1282/';
dstDir = 'val_642x1282_1fold_pad1/';
mkdir(dstDir)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    im_padded1 = padarray(im,[190,382],'pre');
    im_padded1 = padarray(im_padded1,[192,384],'post');
    imwrite(uint8(im_padded1), [dstDir filelist{i} '.png'],'png');
end






%%%%%%%%%%%%%%%%%%% Masks for Iterative Prediciton 
new_masks = ones(642,1282,1);

new_mask1 = padarray(new_masks,[95,191],'pre');
new_mask1 = padarray(new_mask1,[96,192],'post');
new_mask2 = padarray(new_masks,[64,128],'both');
new_mask3 = padarray(new_masks,[48,96],'both');
imwrite(uint8(new_mask1), 'mask_2fold_pad1.png','png');
imwrite(uint8(new_mask2), 'mask_3fold_pad1.png','png');
imwrite(uint8(new_mask3), 'mask_4fold_pad1.png','png');

new_mask1_p2 = ones(size(new_mask1));
new_mask2_p2 = ones(size(new_mask2));
new_mask3_p2 = ones(size(new_mask3));

new_mask1_p2 = padarray(new_mask1_p2 ,[95,191],'pre');
new_mask1_p2 = padarray(new_mask1_p2 ,[96,192],'post');

new_mask2_p2 = padarray(new_mask2_p2 ,[63,127],'pre');
new_mask2_p2 = padarray(new_mask2_p2,[64,128],'post');

new_mask3_p2 = padarray(new_mask3_p2,[48,96],'both');

imwrite(uint8(new_mask1_p2), 'mask_2fold_pad2.png','png');
imwrite(uint8(new_mask2_p2), 'mask_3fold_pad2.png','png');
imwrite(uint8(new_mask3_p2), 'mask_4fold_pad2.png','png');

%%%%%
new_mask2_p3 = padarray(new_mask2_p2 ,[63,127],'pre');
new_mask2_p3 = padarray(new_mask2_p3,[64,128],'post');

new_mask3_p3 = padarray(new_mask3_p2,[47,95],'pre');
new_mask3_p3 = padarray(new_mask3_p3,[48,96],'post');

imwrite(uint8(new_mask2_p3), 'mask_3fold_pad3.png','png');
imwrite(uint8(new_mask3_p3), 'mask_4fold_pad3.png','png');
%%%%%%
new_mask3_p4 = padarray(new_mask3_p3,[47,95],'pre');
new_mask3_p4 = padarray(new_mask3_p4,[48,96],'post');

imwrite(uint8(new_mask3_p4), 'mask_4fold_pad4.png','png');

%%%%%%%%%%%%%%%%%% Draw grid on img  
rectangle('Position',[192,96,1665,833],'LineWidth',2,'EdgeColor','w')

rectangle('Position',[254,126,1538,770],'LineWidth',2,'EdgeColor','w')
rectangle('Position',[127,63,1793,897],'LineWidth',2,'EdgeColor','w')

rectangle('Position',[95,47,1857,929],'LineWidth',2,'EdgeColor','w')
rectangle('Position',[190,94,1666,834],'LineWidth',2,'EdgeColor','w')
rectangle('Position',[286,142,1474,738],'LineWidth',2,'EdgeColor','w')

%% opencv
im = imread('/home/garbade/models_tf/05_Cityscapes/22_msc_fullSizeInput_nc19/val_642x1282_1fold_pad1/frankfurt_000000_000294.png');
thickness = 2;
im_c = cv.rectangle(im,[382,190,1282,642],'Thickness',thickness,'Color',[255 255 255]);
imwrite(im_c,'input_1fold.png');

im_b1 = cv.rectangle(im_c,[192,96,1665,833],'Thickness',thickness,'Color',[255 255 255]);
imwrite(im_b1,'input_2fold.png');
im_b2 = cv.rectangle(im_c,[254,126,1538,770],'Thickness',thickness,'Color',[255 255 255]);
im_b2 = cv.rectangle(im_b2,[127,63,1793,897],'Thickness',thickness,'Color',[255 255 255]);
imwrite(im_b2,'input_3fold.png');
im_b3 = cv.rectangle(im_c,[95,47,1857,929],'Thickness',thickness,'Color',[255 255 255]);
im_b3 = cv.rectangle(im_b3,[190,94,1666,834],'Thickness',thickness,'Color',[255 255 255]);
im_b3 = cv.rectangle(im_b3,[286,142,1474,738],'Thickness',thickness,'Color',[255 255 255]);
imwrite(im_b3,'input_4fold.png');

%% Produce result images
im = imread('~/datasets/cityscapes/val_labels_colors/frankfurt_000000_000294.png');
im(8:80:end,:,:) = 255;
im(:,16:80:end,:) = 255;
cv.imwrite('resultImg.png',im)
mask = imread('~/datasets/cityscapes/masks/01_center_visible/mask_sz100_1024x2048.png');
im_masked = im .* repmat(mask,1,1,3);
mask_grid = mask;
mask_grid(8:80:end,:,:) = 255;
mask_grid(:,16:80:end,:) = 255;
mask_grid_inpaint = mask_grid;
mask_grid_inpaint = repmat(mask_grid,1,1,3);
mask_grid_inpaint(190:832,383:1664,:) = im_masked(190:832,383:1664,:);
imshow(mask_grid_inpaint)
cv.imwrite('resultImg.png',im)

mask_grid_inpaint = repmat(mask_grid,1,1,3);
mask_grid_inpaint(249:775,417:1632,:) = im_masked(249:775,417:1632,:);
cv.imwrite('resultImg.png',mask_grid_inpaint)


%%%%% Crop gt images
cd('/home/garbade/models_tf/05_Cityscapes/46_1_DlSg_k1_s1');
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
% im = imread('val_labels_colors/frankfurt_000000_000294.png');

% srcDir = 'val_labels_colors/';
srcDir = 'val_labels_colors_cropped/';
mkdir(dstDir0)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
%     im_cropped0 = im(191:832,383:1664,:);
%     imwrite(uint8(im_cropped0), [dstDir0 filelist{i} '.png'],'png');
end

% Create multiple crops of gt as input for padding
srcDir = 'val_labels_colors/';
dstDir1 = 'gt_2fold_pad1/';
dstDir2 = 'gt_3fold_pad1/';
dstDir3 = 'gt_3fold_pad2/';
dstDir4 = 'gt_4fold_pad1/';
dstDir5 = 'gt_4fold_pad2/';
dstDir6 = 'gt_4fold_pad3/';

mkdir(dstDir1)
mkdir(dstDir2)
mkdir(dstDir3)
mkdir(dstDir4)
mkdir(dstDir5)
mkdir(dstDir6)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    gt_2f_1 = im(191-95:832+96,383-191:1664+192,:);

    gt_3f_1 = im(191-64:832+64,383-128:1664+128,:);
    gt_3f_2 = im(191-64-63:832+64+64,383-128-127:1664+128+128,:);

%     gt_4f_1 = im(191-48:832+48,383-96:1664+96,:);
%     gt_4f_2 = im(191-48-47:832+48+48,383-96-96:1664+96+96,:);
%     gt_4f_3 = im(191-48-47-47:832+48+48,383-96-96-95:1664+96+96+96,:);
    
    imwrite(uint8(gt_2f_1), [dstDir1 filelist{i} '.png'],'png');
    imwrite(uint8(gt_3f_1), [dstDir2 filelist{i} '.png'],'png');
    imwrite(uint8(gt_3f_2), [dstDir3 filelist{i} '.png'],'png');
%     imwrite(uint8(gt_4f_1), [dstDir4 filelist{i} '.png'],'png');
%     imwrite(uint8(gt_4f_2), [dstDir5 filelist{i} '.png'],'png');
%     imwrite(uint8(gt_4f_3), [dstDir6 filelist{i} '.png'],'png');
    
end

    %% first padding for successive prediction
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
% srcDir = 'val_642x1282/';
% srcDir = 'val_semseg_642x1282/';
srcDir = 'val_labels_colors_cropped/';
dstDir0 = 'gt_colors_642x1282_1fold_pad1/';
dstDir1 = 'gt_colors_642x1282_2fold_pad1/';
dstDir2 = 'gt_colors_642x1282_3fold_pad1/';
dstDir3 = 'gt_colors_642x1282_4fold_pad1/';
mkdir(dstDir0)
mkdir(dstDir1)
mkdir(dstDir2)
mkdir(dstDir3)
for i = 1:numImgs
    im = imread([srcDir filelist{i} '.png'] );
    % 1fold
    im_padded0 = padarray(im,[190,382],'pre');
    im_padded0 = padarray(im_padded0,[192,384],'post');  
    imwrite(uint8(im_padded0), [dstDir0 filelist{i} '.png'],'png');    
    % 2fold
    im_padded1 = padarray(im,[95,191],'pre');
    im_padded1 = padarray(im_padded1,[96,192],'post');
    imwrite(uint8(im_padded1), [dstDir1 filelist{i} '.png'],'png');
    % 3fold
    im_padded2 = padarray(im,[64,128],'both');
    imwrite(uint8(im_padded2), [dstDir2 filelist{i} '.png'],'png');
    % 4fold
    im_padded3 = padarray(im,[48,96],'both');
    imwrite(uint8(im_padded3), [dstDir3 filelist{i} '.png'],'png');    
end
%% second padding for successive prediction
% srcDir = 'val_642x1282/';
srcDir1 = 'gt_2fold_pad1/';
srcDir2 = 'gt_3fold_pad1/';
srcDir3 = 'gt_4fold_pad1/';

dstDir1 = 'gt_2fold_pad1_padded/';
dstDir2 = 'gt_3fold_pad1_padded/';
dstDir3 = 'gt_4fold_pad1_padded/';
mkdir(dstDir1)
mkdir(dstDir2)
mkdir(dstDir3)
for i = 1:numImgs
    % 2fold
    im = imread([srcDir1 filelist{i} '.png'] );
    im_padded1 = padarray(im,[95,191],'pre');
    im_padded1 = padarray(im_padded1,[96,192],'post');
    imwrite(uint8(im_padded1), [dstDir1 filelist{i} '.png'],'png');
    % 3fold
    im = imread([srcDir2 filelist{i} '.png'] );
    im_padded2 = padarray(im,[63,127],'pre');
    im_padded2 = padarray(im_padded2,[64,128],'post');
    imwrite(uint8(im_padded2), [dstDir2 filelist{i} '.png'],'png');
    % 4fold
    im = imread([srcDir3 filelist{i} '.png'] );
    im_padded3 = padarray(im,[48,96],'both');
    imwrite(uint8(im_padded3), [dstDir3 filelist{i} '.png'],'png');    
end
%% third padding for successive prediction
srcDir2 = 'gt_3fold_pad2/';
srcDir3 = 'gt_4fold_pad2/';
dstDir2 = 'gt_3fold_pad2_padded/';
dstDir3 = 'gt_4fold_pad2_padded/';
mkdir(dstDir2)
mkdir(dstDir3)
for i = 1:numImgs
    % 3fold
    im = imread([srcDir2 filelist{i} '.png'] );
    im_padded2 = padarray(im,[63,127],'pre');
    im_padded2 = padarray(im_padded2,[64,128],'post');
    imwrite(uint8(im_padded2), [dstDir2 filelist{i} '.png'],'png');
    % 4fold
    im = imread([srcDir3 filelist{i} '.png'] );
    im_padded3 = padarray(im,[47,95],'pre');
    im_padded3 = padarray(im_padded3,[48,96],'post');  
    imwrite(uint8(im_padded3), [dstDir3 filelist{i} '.png'],'png');    
end

%% fourth padding for successive prediction
srcDir3 = 'gt_4fold_pad3/';
dstDir3 = 'gt_4fold_pad3_padded/';
mkdir(dstDir3)
for i = 1:numImgs
    % 4fold
    im = imread([srcDir3 filelist{i} '.png'] );
    im_padded3 = padarray(im,[47,95],'pre');
    im_padded3 = padarray(im_padded3,[48,96],'post');  
    imwrite(uint8(im_padded3), [dstDir3 filelist{i} '.png'],'png');    
end    
    
% 
%      im_padded1 = padarray(im_cropped,[190,382],'pre');
%      im_padded1 = padarray(im_padded1,[192,384],'post');
% 
% rectangle('Position',[192,96,1665,833],'LineWidth',2,'EdgeColor','w')
% rectangle('Position',[254,126,1538,770],'LineWidth',2,'EdgeColor','w')
% rectangle('Position',[127,63,1793,897],'LineWidth',2,'EdgeColor','w')
% rectangle('Position',[95,47,1857,929],'LineWidth',2,'EdgeColor','w')
% rectangle('Position',[190,94,1666,834],'LineWidth',2,'EdgeColor','w')
% rectangle('Position',[286,142,1474,738],'LineWidth',2,'EdgeColor','w')
% 
% im_c = cv.rectangle(im,[382,190,1282,642],'Thickness',thickness,'Color',[255 255 255]);
% im_b1 = cv.rectangle(im_c,[192,96,1665,833],'Thickness',thickness,'Color',[255 255 255]);
% im_b2 = cv.rectangle(im_c,[254,126,1538,770],'Thickness',thickness,'Color',[255 255 255]);
% im_b2 = cv.rectangle(im_b2,[127,63,1793,897],'Thickness',thickness,'Color',[255 255 255]);
% im_b3 = cv.rectangle(im_c,[95,47,1857,929],'Thickness',thickness,'Color',[255 255 255]);
% im_b3 = cv.rectangle(im_b3,[190,94,1666,834],'Thickness',thickness,'Color',[255 255 255]);
% im_b3 = cv.rectangle(im_b3,[286,142,1474,738],'Thickness',thickness,'Color',[255 255 255]);


%% Qualitative Desults F1 Measure
% box = reshape(1:20,[4,5]);
% cell = imresize(box,[80,80],'nearest');
% imagesc(cell);
% grid = repmat(cell,12,25);
% 
% mask_grid(32:80:end,:,:) = 255;
% mask_grid(:,24:80:end,:) = 255;
% % mask_grid_inpaint(190:832,383:1664,:) = im_masked(190:832,383:1664,:);
% imshow(mask_grid_inpaint)
% mask = zeros(1024,2048,3);
% center = mask(33:1002,24:2023);
% box_0 = box -1;
% box_col = ind2rgb(box_0,colors);
% cell_col = imresize(box_col,[80,80],'nearest');
% boxIdx = reshape(1:12*25,12,25);
% cell_Idx = imresize(linIdx,[800,800],'nearest');

box = reshape(1:20,[4,5]);
cell = imresize(box,[80,80],'nearest');
imagesc(cell);
grid = repmat(cell,12,25);
box_big = imresize(box,[80,80],'nearest');
targets_big = repmat(box_big,12,25);
targets_per_label = false(960,2000,19);
for i = 1:19
    targets_per_label(:,:,i) = targets_big == i;
end


TP_cropped = TP(1:12,1:25,:);
Pred_cropped = pred_pooled{1}(1:12,1:25,:);
Gt_cropped = gt_pooled(1:12,1:25,:);

% TP_cropped = TP(1:12,1:25,:);
Pred_cropped = pred_pooled_raw(1:12,1:25,:);
Gt_cropped = gt_pooled(1:12,1:25,:);

TP_big = imresize(TP_cropped,[960,2000],'nearest');
overlay = targets_per_label & TP_big;
for i = 1:19
    imagesc(overlay(:,:,i));pause(1);
    title(classes{i})
end
overlay_final  = max(overlay,[],3);

labelsFile = '/home/garbade/datasets/cityscapes/labels.mat';
colMap = load(labelsFile);
colorNames = colMap.colorNames; %% TODO: load labels
colors = colMap.colors;

targets_big_col = ind2rgb(targets_big,colors);
result = targets_big_col .* repmat(overlay_final,1,1,3);


%% Compute deloc results
TP_result = displayDelocalisedLabels(TP_cropped);
Pred_result = displayDelocalisedLabels(Pred_cropped);
Gt_result = displayDelocalisedLabels(Gt_cropped);

%% Store results
im = imread('~/datasets/cityscapes/val_labels_colors/frankfurt_000000_000294.png');
im_padded = zeros(size(im));
% Convert into image with black surrounding / padding and center crop of 642x1282
im_padded(191:832,383:1664,:) = im(191:832,383:1664,:);
im_960x2000 = im_padded(33:992,24:2023,:);

Gt_combined = Gt_result;
 
Gt_combined(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3) = im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);
cv.imwrite([imname '_result_Gt_pool10x10.png'],uint8(Gt_combined));

Pred_combined = Pred_result;
Pred_combined(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3)= im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);

cv.imwrite([imname '_result_Pred_pool10x10.png'],uint8(Pred_combined));


%% With white cell borders
Gt_combined_cells = Gt_combined;
Pred_combined_cells = Pred_combined;

Gt_combined_cells(80:80:end,:,:) = 255;
Gt_combined_cells(:,80:80:end,:) = 255;
Gt_combined_cells(81:80:end,:,:) = 255;
Gt_combined_cells(:,81:80:end,:) = 255;
Gt_combined_cells(79:80:end,:,:) = 255;
Gt_combined_cells(:,79:80:end,:) = 255;
Gt_combined_cells(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3) = im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);
Pred_combined_cells(80:80:end,:,:) = 255;
Pred_combined_cells(:,80:80:end,:) = 255;
Pred_combined_cells(81:80:end,:,:) = 255;
Pred_combined_cells(:,81:80:end,:) = 255;
Pred_combined_cells(79:80:end,:,:) = 255;
Pred_combined_cells(:,79:80:end,:) = 255;
Pred_combined_cells(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3) = im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);

cv.imwrite([imname '_result_Gt_pool10x10_cells3.png'],uint8(Gt_combined_cells));
cv.imwrite([imname '_result_Pred_pool10x10_cells3.png'],uint8(Pred_combined_cells));


%% Display masks for debugging
figure(2);imagesc(ic_pooled(:,:,1))
figure(3);imagesc(mask_pooled(:,:,1))
% total_mask = ic_pooled(2:end,2:end,:) | mask_pooled(2:end,2:end,:);
total_mask = ic_pooled & mask_pooled;
figure(1);imagesc(total_mask(:,:,1))

%% Border replication
cd('~/models_tf/5_Cityscapes/56_1_BorderReplicate_gt/')
filelist = getNamesFromAsciiFile('~/deeplab_tf/11_fixMskAndDelocLoss/dataset/city/val_id.txt');
numImgs = size(filelist,1);
% % srcDir = '/home/garbade/datasets/cityscapes/val_labels_colors_cropped/';
% srcDir = '/home/garbade/datasets/cityscapes/labels_ic19_val_642x1282/';
% % srcDir = '/home/garbade/models_tf/05_Cityscapes/22_msc_fullSizeInput_nc19/val_semseg_642x1282_ind/';

srcDir = 'val_predNetCol_1fold_pad1_prob/';

dstDir0 = 'val_predNetCol_1fold_pad1_prob_border_repeat/';
mkdir(dstDir0)
for i = 1:numImgs
%     im = imread([srcDir filelist{i} '.png'] );
    im = load([srcDir filelist{i} '.mat'] );
    im = im.data;
    % 1fold
    im_padded0 = padarray(im,[23,47],'pre','replicate');
    im_padded0 = padarray(im_padded0,[24,48],'post','replicate');  
    data = im_padded0;
    %imwrite(uint8(im_padded0), [dstDir0 filelist{i} '.png'],'png');    
    save([dstDir0 filelist{i} '.mat'],'data');  
end


%% Export delocalized images
% Folder to plot:
% /media/data1/models_tf/05_Cityscapes/54_5_kk_DlSg_k10_s10/val_predNetCol_2fold_pad2_prob
% Use Matconvnet CPU version
cd ~/libs/matconvnet/matconvnet-1.0-beta24_CPU/
addpath('matlab')
run matlab/vl_setupnn.m
% Change to result dir
cd('/media/data1/models_tf/05_Cityscapes/54_5_kk_DlSg_k10_s10/')
resDir = 'val_predNetCol_2fold_pad2_prob/';
gtDir = '~/datasets/cityscapes/labels_ic19/';
kernel = 10; 
stride = 10;
mode = 'val_predNetCol_2fold_pad2_prob';
use_ic = 0;
expFolder = '54_5_kk_DlSg_k10_s10';
use_L1_loss = 0;

[acc_all] = city_evalSeg_delocLoss_debug(resDir, ...
                             gtDir, ...
                             kernel, ...
                             stride, ...
                             mode, ...
                             'ignore_void_class',false,...
                             'ExpName',expFolder,...
                             'L1',false); 
                             




















