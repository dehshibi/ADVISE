function classScore = iGetClassScoreAtChannelDimension(outputActivations,classIndex)
    %Create a cell array that looks something like {:,:,:,:}
    dynamicIndices = repmat({':'}, 1, ndims(outputActivations));

    %Make a cell array to look like {:,:,c,:} where c is in the
    %position of the channel dimension
    channelDimension = finddim(outputActivations,'C');
    dynamicIndices(channelDimension) = {classIndex};

    %This is equivalent to outputActivations(:,:,classIndex,:)
    classScore = outputActivations(dynamicIndices{:});
end