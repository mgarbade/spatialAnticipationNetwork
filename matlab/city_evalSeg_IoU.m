function [pixel_acc,class_acc,IoU,conf] = city_evalSeg_IoU(resDir,...
                                                          gtDir,...
                                                          varargin)
% city_evalSeg Evaluates a set of segmentation results.
% [pixel_acc,class_acc,IoU,conf] = city_evalSeg(resDir,gtDir,varargin) prints out the per class and overall
% segmentation accuracies. Accuracies are given using the intersection/union (IoU)
% metric:
%   true positives / (true positives + false positives + false negatives) 
% returns class_acc: the per class percentage ACCURACIES, the average accuracy AVACC and the confusion
% matrix conf.
% the unnormalised confusion matrix, which contains raw pixel counts.


opt = struct('MaskMode',3,...  % 1..3  eval on --> fullImg, visibleArea, occludedArea
             'n_classes',19,...
             'ignore_label',19,... % Everything below this label will be ignored
             'ExpName','ExpName',...
             'MaskFile','/home/garbade/BMVC2017/code/mask/mask_642x1282.png'); 
opt = processInputArgs(opt, varargin{:});
gtids = getNamesFromAsciiFile('~/datasets/cityscapes/val_id.txt');

postfix = '';
classes = {'road' 'sidewalk' 'building' 'wall' 'fence' 'pole' 'traffic light' 'traffic sign' ...
    'vegetation' 'terrain' 'sky' 'person' 'rider' 'car' 'truck' 'bus' 'train' 'motorcycle' 'bicycle' 'void'};
mask = imread(opt.MaskFile);

% number of labels = number of classes plus one for the background
nclasses = opt.n_classes;
num = opt.n_classes; % actual number of classes -> used to create confusion matrix
ignore_label = opt.ignore_label;
confcounts = zeros(num);
count = 0;
num_missing_img = 0;

tic;
for i=1:length(gtids)
    imname = gtids{i};
    
    % ground truth label file
    gtfile = [gtDir '/' imname postfix '.png'];
    [gtim,map] = imread(gtfile);    
    gtim = double(gtim);
    
    % results file
    resfile = [resDir '/' imname '.png'];
    try
      [resim,map] = imread(resfile);
    catch err
      num_missing_img = num_missing_img + 1;
      continue;
    end

    resim = double(resim);
    
    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel > nclasses), 
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,nclasses);
    end

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim ~= szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    
    locs = gtim < ignore_label;% Ignore void label

    
    %% Different evaluation areas
    switch opt.MaskMode
        case 1
            mode = 'fullImg';
        case 2
            mode = 'visibleArea';
            locs(mask == 0) = false;
        case 3
            % occluded area
            mode = 'occludedArea';
            locs(~mask == 0) = false; 
    end
    
    
    % joint histogram
    sumim = 1 + gtim + resim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);
end

if (num_missing_img > 0)
  fprintf(1, 'WARNING: There are %d missing results!\n', num_missing_img);
end

% confusion matrix - first index is true label, second is inferred label
conf = 100 * confcounts ./ repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
rawcounts = confcounts;

% Pixel Accuracy
pixel_acc = 100 * sum(diag(confcounts)) / sum(confcounts(:));

% Class Accuracy
class_accs = zeros(1, num);
class_count = 0;
for i = 1 : num
    denom = sum(confcounts(i, :));
    if (denom == 0)
        denom = 1;
    end
    class_accs(i) = 100 * confcounts(i, i) / denom; 
    clname = classes{i};
    if ~strcmp(clname, 'void')
        class_count = class_count + 1;
        % fprintf('  %14s: %6.3f%%\n', clname, class_accs(i));
    end
end
class_acc = sum(class_accs) / num;

% Pixel IOU
accuracies = zeros(nclasses,1);

real_class_count = 0;
for j = 1:num
   
   gtj = sum(confcounts(j,:));
   resj = sum(confcounts(:,j));
   gtjresj = confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative) 
   % which is equivalent to the following percentage:
   denom = (gtj+resj-gtjresj);

   if denom == 0
     denom = 1;
   end
   
   accuracies(j)=100*gtjresj/denom;
   clname = classes{j};
   real_class_count = real_class_count + 1;
   if ~strcmp(clname, 'void')
   end
end


IoU = sum(accuracies) / real_class_count;

resFilename = [resDir '/res_' datestr(now, 'yyyy-mm-dd-HH-MM-SS') '_' mode '.mat'];
save(resFilename,'confcounts','pixel_acc','class_accs','class_acc','accuracies','IoU');

resString = ['PixelAcc= ' num2str(pixel_acc) ' ClassAcc= ' num2str(class_acc) '  IoU= ' num2str(IoU) ];
disp(resString)

resString = [ datestr(now, 'yyyy-mm-dd-HH-MM-SS') ', ExpName:, ' opt.ExpName ', mode:, ' mode ', ' resDir ',  IoU =, ' num2str(IoU) ];

% Write to master log-file
system(['echo "' resString '" >> ~/resultsPixelAcc_IoU.txt']);












