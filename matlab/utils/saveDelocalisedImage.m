function saveDelocalisedImage(Pred_cropped,saveDir,resImgDir, imname)


%% Compute deloc results
Pred_result = displayDelocalisedLabels(Pred_cropped);

%% Store results
% Convert into image with black surrounding / padding and center crop of 642x1282
im = imread([resImgDir imname '.png']);
im_padded = zeros(size(im));
im_padded(191:832,383:1664,:) = im(191:832,383:1664,:);
im_960x2000 = im_padded(33:992,24:2023,:);

Pred_combined = Pred_result;
Pred_combined(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3)= im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);

%% With white cell borders
Pred_combined_cells = Pred_combined;
Pred_combined_cells(80:80:end,:,:) = 255;
Pred_combined_cells(:,80:80:end,:) = 255;
Pred_combined_cells(81:80:end,:,:) = 255;
Pred_combined_cells(:,81:80:end,:) = 255;
Pred_combined_cells(79:80:end,:,:) = 255;
Pred_combined_cells(:,79:80:end,:) = 255;
Pred_combined_cells(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3) = im_960x2000(3*80+1:960-3*80-1,5*80+1:2000-5*80-1,1:3);

imwrite(uint8(Pred_combined_cells),[saveDir '/' imname '.png']);