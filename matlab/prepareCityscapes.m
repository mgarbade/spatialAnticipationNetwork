% Convert label index files
% Switch from 255 to 19
filelist = getNamesFromAsciiFile('val_id.txt');
numImgs = size(filelist,1);
for i = 1:numImgs
    gtLabel = imread(['labels/' filelist{i} '.png']);
    gtLabel(gtLabel == 255) = 19;
    
    imwrite(uint8(gtLabel),['labels_ic19/' filelist{i} '.png'],'png');
end
    
