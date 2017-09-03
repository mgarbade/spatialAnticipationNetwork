function [acc_all] = city_evalSeg_F1(resDir,...
                                                  gtDir,...
                                                  krnl,...
                                                  stride,...
                                                  mode,...
                                                  varargin)

% Padding according to kernel_size                                                      
start = tic;
p = floor(krnl/2);                                                          
thresholds = [0.5];
opt = struct('MaskMode',1,...  % 1..3  eval on --> fullImg, ImgWithoutBB, BBonly
             'sz',50, ...      % 1..4  apply on img with mean bar at [l r t b] --> [left right top bottom]
             'n_classes',19,...
             'ExpName','ExpName',...
             'L1',false);      % --> set this to true for experiments trained with L1 loss

opt = processInputArgs(opt, varargin{:});
gtids = getNamesFromAsciiFile('~/datasets/cityscapes/val_id.txt');
postfix = '';
classes = {'road' 'sidewalk' 'building' 'wall' 'fence' 'pole' 'traffic light' 'traffic sign' ...
    'vegetation' 'terrain' 'sky' 'person' 'rider' 'car' 'truck' 'bus' 'train' 'motorcycle' 'bicycle' 'void'};

MASK_FILE = '/home/garbade/BMVC2017/code/mask/mask_642x1282.png';

% Save qualitative results
fullExpPath = ['/home/garbade/models_tf/05_Cityscapes/' opt.ExpName];
saveDirGt = [fullExpPath '/' mode '_DelocGt'];
saveDirPred = [fullExpPath '/' mode '_DelocPred'];    
mkdir(saveDirGt)
mkdir(saveDirPred)
resImgDirGt = '~/datasets/cityscapes/val_labels_colors/';
resImgDirPred = [fullExpPath '/' mode(1:end-5) '/'];

% number of labels = number of classes plus one for the background (20 + 1)
n_classes = opt.n_classes;

% Mask
mask = imread(MASK_FILE);
mask_outside = ~mask;
mask_outside_rep = repmat(mask_outside,1,1,n_classes);

numImgs = length(gtids);
numThresholds = length(thresholds);

TP_total = cell(numThresholds,1);
Pred_total = cell(numThresholds,1);
Gt_total = cell(numThresholds,1);

tic;
for i = 1:numImgs    
    
    % display progress
    if toc > 60
        fprintf('test confusion: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
    imname = gtids{i};
    % ground truth label file
    gtfile = [gtDir '/' imname postfix '.png'];
    [gtim,map] = imread(gtfile);    
    gtim = double(gtim);
    % results file
    resfile = [resDir '/' imname '.mat'];
    resim = load(resfile);
    pred_one_hot = resim.data;
    gt_one_hot = zeros([size(pred_one_hot,1),size(pred_one_hot,2),n_classes+1] );
    for j = 1:n_classes+1 % The +1 dimension is the void label
        gt_ind = ind2sub(gtim, gtim == j-1);
        gt_one_hot(:,:,j) = imresize(gt_ind,128/1024,'nearest');
    end

    pred_big = pred_one_hot;
    gt_pooled_raw = vl_nnpool(single(gt_one_hot),[krnl,krnl],...
                                            'Pad',[p,p,p,p],...
                                            'Stride',stride);
	gt_pooled_raw = logical(gt_pooled_raw);
    mask_pooled = imresize(mask_outside_rep,[size(gt_pooled_raw,1),size(gt_pooled_raw,2)],'nearest');
    mask_pooled = logical(mask_pooled);
    ic_pooled = repmat(gt_pooled_raw(:,:,n_classes+1),1,1,n_classes);
    ic_pooled = ~ic_pooled;
    
    gt_pooled_raw = gt_pooled_raw(:,:,1:n_classes);
    % Save qualitative results
    Gt_cropped = gt_pooled_raw(2:13,2:26,:);
    saveDelocalisedImage(Gt_cropped,saveDirGt,resImgDirGt, imname)   
    
    % Mask with inside mask
    gt_pooled = gt_pooled_raw & mask_pooled;
    
    if opt.L1
        % Use argmax for labels
       pred_softmax = vl_nnsoftmax(pred_big);
       [~,pred_argmax]= max(pred_softmax,[],3);
       pred_big_prob = zeros(size(pred_softmax));
        for j = 1:n_classes 
            pred_big_prob(:,:,j) = ind2sub(pred_argmax,pred_argmax==j);
        end
    else
       pred_big_sigmoid = vl_nnsigmoid(pred_big);
       pred_big_prob = pred_big_sigmoid;
    end
    pred_pooled = cell(numThresholds,1);
    for j = 1:numThresholds
        thresh = thresholds(j);
        pred_big_thresh = pred_big_prob > thresh;
        pred_pooled_raw = vl_nnpool(single(pred_big_thresh),[krnl,krnl],...
                                                'Pad',[p,p,p,p],...
                                                'Stride',stride);
        pred_pooled_raw = logical(pred_pooled_raw);
        
        Pred_cropped = pred_pooled_raw(2:13,2:26,:);
        saveDelocalisedImage(Pred_cropped,saveDirPred,resImgDirPred, imname)   
        
        % Mask prediction
        pred_pooled{j} = pred_pooled_raw & mask_pooled;  
        
        % Get TP
        TP = pred_pooled{j} & gt_pooled;
        TP = squeeze(sum(sum(TP,1),2)+eps);
        PD = squeeze(sum(sum(pred_pooled{j},1),2)+eps);
        GT = squeeze(sum(sum(gt_pooled,1),2)+eps);
        
        if i == 1 
            TP_total{j} = TP;
            Pred_total{j} = PD;
            Gt_total{j} = GT;
        else
            TP_total{j} = TP_total{j} + TP;
            Pred_total{j} = Pred_total{j} + PD;
            Gt_total{j} = Gt_total{j} + GT;            
        end
        


    end
end

mPrecision = zeros(1,numThresholds);
mRecall = zeros(1,numThresholds);
mF1 = zeros(1,numThresholds);

for j = 1:numThresholds
    Precision_perClass = TP_total{j}./ Pred_total{j};
    mPrecision(j) = 100 * mean(Precision_perClass);
    Recall_perClass = TP_total{j}./ Gt_total{j};
    mRecall(j) = 100 * mean(Recall_perClass);
    F1_perClass = (2 * Precision_perClass .* Recall_perClass +eps)  ./ (Precision_perClass + Recall_perClass + eps);
    mF1(j) = 100 * mean(F1_perClass);
end


resString = [ datestr(now, 'yyyy-mm-dd-HH-MM-SS') ',ExpName:, ' opt.ExpName ', ResDir:, ' resDir ', krnl=,' num2str(krnl) ...
             ' , stride =, ' num2str(stride) ...
             ' , Precision =, ' num2str(mPrecision) ...
             ' , Recall =, ' num2str(mRecall) ...
             ' , F1 =, ' num2str(mean(mF1,1)) ...
             ' , Thresholds =, ' num2str(thresholds)];
acc_all = mean(mF1,1);
disp(resString);
displayElapsedTime(start);
system(['echo "' resString '" >> ' mode 'F1_07.txt']);
% Write to master log-file
system(['echo "' resString '" >> ~/resultsDelocLoss.txt']);



