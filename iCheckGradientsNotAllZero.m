function iCheckGradientsNotAllZero(dScoresdMap)
% If all gradients are zero, map will be zero. This most likely happens
% when the feature layer is incorrectly specified as downstream from
% the reduction layer, so there is no tracing between classScore and
% featureMap.

if all(dScoresdMap == 0)
    error(message("AllGradientsZero"));
end   
end