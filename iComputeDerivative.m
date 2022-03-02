function [featureMap,dScoresdFeatures] = iComputeDerivative(net,data,classification,reductionFunction,reductionLayer,featureLayer,classIndex)
    %Get the class scores and the feature map from the network
    [scores,featureMap] = predict(net, data, 'Outputs',...
    {reductionLayer, featureLayer});
    
    %If the user has provided a class as opposed to a reduction function
    if ~isempty(classification)
        %If the output activations have a spatial dimension, further
        %reduction is needed to reduce this to a scalar. Sum across spatial
        %dimensions
        %This occurs when the network is being used for semantic
        %segmentation, see arXiv:2002.11434v1
        if ~isempty(finddim(scores,'S'))
            outputActivations = sum(scores,finddim(scores,'S'));
        else
            outputActivations = scores;
        end
        
        %Get the class at the channel dimension. We don't know which
        %dimension this is (it could differ for 2D vs 3D), so we have to
        %work this out dynamically
        classScore = iGetClassScoreAtChannelDimension(outputActivations,classIndex);
        
        dScoresdFeatures = dlgradient(classScore(1), featureMap);
        iCheckGradientsNotAllZero(dScoresdFeatures);
        
        % If classIndex isn't scalar, the user has asked for maps for multiple
        % classes. Compute the gradient for each class score separately and
        % concatenate along an added final dimension. This will give an
        % SSCU dlarray for 2D image data.
        %
        % We can't easily concatenate dlarrays repeatedly in a loop due to
        % dimension labelling; put them in a cell array and do a single
        % concatenation at the end.
        if length(classIndex) > 1
            
            % Cell array of e.g. SSC dlarrays
            dScoresdFeaturesCell = {dScoresdFeatures};
            
            for i=2:length(classIndex)
                dScoresdFeaturesCell{i} = dlgradient(classScore(i), featureMap);
                iCheckGradientsNotAllZero(dScoresdFeaturesCell{i});
            end
            
            % Single SSCU dlarray.
            dScoresdFeatures = cat(ndims(dScoresdFeatures) + 1, dScoresdFeaturesCell{:});
        end

    else
        classScore = reductionFunction(scores);
        
        %If classScore isn't a scalar value, we can't use it to differentiate
        %the activations of the feature layer. Throw an error.
        if ~isscalar(classScore)
            %Throw ActivationsDoNotReduceToScalar if the user has provided
            %a reduction function, but it doesn't reduce activations
            error(message("Activations don't reduce to scalar"));
        end
        
        dScoresdFeatures = dlgradient(classScore, featureMap);
        iCheckGradientsNotAllZero(dScoresdFeatures);
    end
end
