function displayImgWithOverlay(img, ... 
                               labelImg, ... 
                               colorNames, ...
                               colors, ...
                               backgroundInvisible, ...
                               showText)

    if nargin < 5 
        backgroundInvisible = false;
    end
    if nargin < 6
        showText = false;
    end
    
    imshow(img);
    hold on;
    overlay_final = ind2rgb(labelImg,colors);
    overlay_final = uint8(overlay_final);
    hfig = imshow(overlay_final);
    
    labelImg = double(labelImg);
    
    if backgroundInvisible          % In case of Pascal VOC -> don't display background class
        fullMask = im2bw(labelImg);
        fullMask = double(fullMask);
        set(hfig,'AlphaData',fullMask* 0.5);
    else
        set(hfig,'AlphaData', 0.5);
    end
    
    numClasses = size(colorNames,1);

    % Only use if text is wanted
    if showText
        for j = 1:numClasses % Start with 2 because background has no color
            BW = (labelImg == j - 1);
            regions = regionprops(BW,'PixelIdxList','Centroid','Area','BoundingBox');
            numComponents = size(regions,1);
            if numComponents == 0
                continue
            end
            areas = zeros(numComponents,1);
            centroids = zeros(numComponents,2);
            for k = 1:numComponents
                areas(k,1) = regions(k,1).Area;
                centroids(k,:) = regions(k,1).Centroid;
            end
            % Mark only biggest region with text info
            [~,ind] = max(areas);
            text(centroids(ind,1), centroids(ind,2), colorNames{j});
        end
    end
    hold off;
    drawnow;
    pause(0.005);