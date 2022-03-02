function gradcamMap = iComputeSum(featureMap, dScoresdMap, dlData, Z)
%Assume the user only has spatial and channel dimensions for now
%     Z = size(featureMap,1)*size(featureMap,2);
    tmp = sum(dScoresdMap, finddim(dlData,'S'))/Z;
    gradcamMap = sum(featureMap .* tmp, finddim(dlData,'C'));
%     gradcamMap = sum(featureMap .* sum(dScoresdMap, ...
%         finddim(dlData,'S')), finddim(dlData,'C'));
    gradcamMap = relu(gradcamMap);
    gradcamMap = extractdata(gradcamMap);

    % If the user provided multiple class labels, dScoresdMap will
    % be SSCU (for 2D image data) with non-singleton U
    % corresponding to 1 entry per class label. The above sum will
    % work as intended and leave us an SSCU array where C is
    % singleton but U is not. Squeeze so that we get rid of the C
    % dim and return SSU.
    gradcamMap = squeeze(gradcamMap);
end