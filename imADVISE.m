function [evaluationMetrics,lableMax] = imADVISE(net,dlnet,inputSize,inputImage,...
    featureLayer,reductionLayer,reductionFunction,tags,displayFlag)
%% Load input(s) and format as and DLARRAY.
%-- Pay attention to use proper cobmination on 'S' and 'C' based on the
%   input type of the network. Otherwise, 'PREDICT' function won't work.
img = inputImage;
img = imresize(img,inputSize);
data = single(img);
data = dlarray(data,'SSC');
cnt = 1;
%% Classification
[scores,featureMap] = predict(dlnet, data, 'Outputs',...
    {reductionLayer, featureLayer});
tmpScores = extractdata(scores);
tmpFeatureMap = extractdata(featureMap);
featureMapSize = size(featureMap);
Z = featureMapSize(1)*featureMapSize(2);
%-- We extract top-k prediction to evaluate Class Sensitivity
k = 2;
[classScore,classIndex] = maxk(tmpScores,k);
classScoreMax = classScore(1);
classIndexMax = classIndex(1);
lableMax = tags(classIndexMax);

classScoreMin = classScore(k);
classIndexMin = classIndex(k);
lableMin = tags(classIndexMin);

%% Calculate ABK
binNumber = linspace(0,1,Z);
kernelRepeat = 10;
peakNumber = zeros(1,featureMapSize(3));
tic
parfor ii=1:featureMapSize(3)
    inputBlock = tmpFeatureMap(:,:,ii);
    inputBlock = inputBlock(:);
    AKBoutput = ADVISEkernel(inputBlock,binNumber,kernelRepeat);
    if ~isempty(AKBoutput)
        tmp_pks = findpeaks(AKBoutput);
        if ~isempty(tmp_pks)
            peakNumber(1,ii) = numel(tmp_pks);
        end
    end
end
timeABK = toc;

%% Calculate gradient
%-- This section calculates the gradient of the output class score
%   with respect to the feature map extracted from 'featureLayer'
[~,dScoresdMapMAX] = dlfeval(@iComputeDerivative,dlnet,data,...
    lableMax,reductionFunction,reductionLayer,featureLayer,classIndexMax);
[~,dScoresdMapMIN] = dlfeval(@iComputeDerivative,dlnet,data,...
    lableMin,reductionFunction,reductionLayer,featureLayer,classIndexMin);

%% Evaluation Metrics.
tiledlayout('flow','Padding','compact')
nexttile
imshow(img,'InitialMagnification',200)
title('\fontsize{16}Input image')

for ii=min(peakNumber):max(peakNumber)
    %-- This section remove channels from 'featureMap' and 'dScoresdMap' that
    %   don't satisfy the threshold with respect to 'peakNumber'.
    reducedFeatureMap = featureMap;
    reducedDScoresdMapMAX = dScoresdMapMAX;
    reducedDScoresdMapMIN = dScoresdMapMIN;
    %-- Apply threshold
    [~,reductionIndx] = find(peakNumber~=ii);
    %-- Remove indexes
    reducedFeatureMap(:,:,reductionIndx)=[];
    reducedDScoresdMapMAX(:,:,reductionIndx)=[];
    reducedDScoresdMapMIN(:,:,reductionIndx)=[];
    
    %-- This section calculate the main visualization operation.
    %   For the case of Grad-CAM, we need to calculate the weighted sum.
    %-- Maximum predicted activation class map.
    rawMapMAX = iComputeSum(reducedFeatureMap,reducedDScoresdMapMAX,...
        data,Z);
    rawMapMAX = rescale(rawMapMAX);
    overlayMapMAX = imresize(rawMapMAX, inputSize, 'Method', 'bicubic');
    %-- Minimum predicted activation class map.
    rawMapMIN = iComputeSum(reducedFeatureMap,reducedDScoresdMapMIN,...
        data,Z);
    rawMapMIN = rescale(rawMapMIN);
    overlayMapMIN = imresize(rawMapMIN, inputSize, 'Method', 'bicubic');
    if size(reducedFeatureMap,3)>0 && sum(rawMapMAX,'all')>0
        metrics(cnt) = iCalculateCAMmetrics(net,img,overlayMapMAX,...
            overlayMapMIN,classScoreMax,classIndexMax,timeABK,...
            size(reducedFeatureMap,3),ii,0);
        cnt = cnt +1;
        nexttile
        imshow(img,'InitialMagnification',200)
        hold on
        imagesc(overlayMapMAX,'AlphaData',0.5)
        colormap jet
        title({['\fontsize{16}ADVISE: ',...
            num2str(size(reducedFeatureMap,3))];...
            ['\fontsize{16}#peak: ', num2str(ii)]})
    end
end
[metrics.time] = deal(timeABK/size(metrics,2));

%% Other Visualisation Methods.
if displayFlag
    tic
    scoreGMapMax = gradCAM(net,img,lableMax,'ReductionLayer',...
        reductionLayer,'FeatureLayer',featureLayer);
    scoreGMapMin = gradCAM(net,img,lableMin,'ReductionLayer',...
        reductionLayer,'FeatureLayer',featureLayer);
    GradTime = toc;
    metrics(cnt) = iCalculateCAMmetrics(net,img,scoreGMapMax,scoreGMapMin,...
        classScoreMax,classIndexMax,GradTime,size(featureMap,3),'Grad-CAM',0);
    
    cnt = cnt+1;
    
    numTopFeatures = 5;
    tic
    [LIMEMapMax,LIMEfeatureMap,LIMEfeatureImportance] = imageLIME(net,img,...
        lableMax);
    [~,idx] = maxk(LIMEfeatureImportance,numTopFeatures);
    mask = ismember(LIMEfeatureMap,idx);
    scoreLIMEMapMax = uint8(mask).*img;
    LIMETime = toc;
    [~,LIMEfeatureMap,LIMEfeatureImportance] = imageLIME(net,img,lableMin);
    [~,idx] = maxk(LIMEfeatureImportance,numTopFeatures);
    mask = ismember(LIMEfeatureMap,idx);
    scoreLIMEMapMin = uint8(mask).*img;
    
    metrics(cnt) = iCalculateCAMmetrics(net,img,scoreLIMEMapMax,...
        scoreLIMEMapMin,classScoreMax,classIndexMax,LIMETime,...
        size(featureMap,3),'LIME',1);
    
    %-- Grad-CAM
    nexttile
    imshow(img,'InitialMagnification',200)
    hold on
    imagesc(scoreGMapMax,'AlphaData',0.5)
    colormap jet
    title('\fontsize{16}Grad-CAM')
    %-- LIME
    nexttile
    imshow(img,'InitialMagnification',200)
    hold on
    imagesc(LIMEMapMax,'AlphaData',0.5)
    colormap jet
    title('\fontsize{16}LIME')
end

evaluationMetrics = metrics;
end%function