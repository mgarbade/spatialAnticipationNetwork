function result = displayDelocalisedLabels(TP_cropped)

box = reshape(1:20,[4,5]);
box_big = imresize(box,[80,80],'nearest');
targets_big = repmat(box_big,12,25);
targets_per_label = false(960,2000,19);
for i = 1:19
    targets_per_label(:,:,i) = targets_big == i;
end

TP_big = imresize(TP_cropped,[960,2000],'nearest');
overlay = targets_per_label & TP_big;
overlay_final  = max(overlay,[],3);

% Colorize
labelsFile = '/home/garbade/datasets/cityscapes/labels.mat';
colMap = load(labelsFile);
colors = colMap.colors;
targets_big_col = ind2rgb(targets_big,colors);

result = targets_big_col .* repmat(overlay_final,1,1,3);





