function metrics = iCalculateCAMmetrics(net,inputImage,MapMax,MapMin,...
    classScoreMax,classIndexMax,executeTime,channelNumber,peakCount,flag)
metrics.peakCount = peakCount;
metrics.channelNumber = channelNumber;
%% Class Sensitivity.
%-- It measures the similarity between maps w.r.t the highest ('cmax')
%   and the lowest ('cmin') scores. For measuring similarity, Pearson
%   correlation is used, where good explanation method should have score
%   close to zero or below zero.
check = 0;
if flag
    R = corrcoef(single(MapMax),single(MapMin));
else
    R = corrcoef(MapMax,MapMin);
end
if R(1,2)>=-0.55 && R(1,2)<=0.55
    check = 1;
end
if isnan(R(1,2))
    R(1,2) = 0;
end
metrics.ClassSensitivity = R(1,2);
%% Hit or Miss
%-- If the index of the predicted class for the input image fall in top-5
%   prediction of the model w.r.t the masked image and CHECK, the rest of 
%   metrics will not be penalised.
if flag
    maskedImage = MapMax;
else
    maskedImage = inputImage.*cast(MapMax,'like',inputImage);
end
scoresExplain = predict(net,maskedImage);
[~,top5] = maxk(scoresExplain,5);
hitORmiss = intersect(classIndexMax,top5);

%% Average Drop.
%-- It measures the average percentage drop in confidence for the
%   target class 'c' when the model sees only the explanation map,
% instead of the full image.
%-- 'yc' is the output score for class c when using the full image.
%-- 'oc' the output score when using the explanation map.
yc = classScoreMax;
oc = max(scoresExplain);
metrics.AverageDrop = (max(0,(yc-oc))/yc);
%% SSIM
metrics.ssimValue = ssim(maskedImage,inputImage);
%% FSIM
[~,metrics.fsimValue] = FeatureSIM(maskedImage,inputImage);
%% MSE
% The minimum value for MSE is 0. However, the maximum depends on the
% type of input image. Here, the input is uint8 and the max value in
% each dimension is 255. So, the max value is 255^2
mseValue = immse(maskedImage,inputImage);
minMSE = 0;
maxMSE = 255*255;
metrics.mseValue = rescale(mseValue,'InputMin',minMSE,...
    'InputMax',maxMSE);
%% XAIValue
%-- XAIValue combines all metrics with which direct comparisons between
%   approaches becomes feasible. It is harmonic mean of all metrics.
if isempty(hitORmiss)
    if check
        diff = 1-abs(yc-oc);
        metrics.AverageDrop = (metrics.AverageDrop)*diff;
        metrics.ssimValue = (metrics.ssimValue)*diff;
        metrics.fsimValue = (metrics.fsimValue)*diff;
        metrics.mseValue = (metrics.mseValue)*diff;
    else
        metrics.AverageDrop = 1;
        metrics.ssimValue = 0;
        metrics.fsimValue = 0;
        metrics.mseValue = 1;
    end
end
component(1) = 1/(metrics.ssimValue);
component(2) = 1/(metrics.fsimValue);
component(3) = 1/(1-metrics.mseValue);
component(4) = 1/(1-metrics.AverageDrop);
metrics.XAIValue = numel(component)/(sum(component,'all'));
metrics.time = executeTime;
end