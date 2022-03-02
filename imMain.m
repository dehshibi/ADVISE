%% Clean memory
close all
clear all
delete(gcp('nocreate'))
clc

%% Set path
folder = fileparts(which('imMain.m')); 
addpath(genpath(folder));
imagesFolder = fullfile(folder,'Images');
clear folder
%% Load variable and Network
% -- 'convLayerNumber' is the layer number responsible for performing 
%       Convolution. 
% -- 'reductionLayerNumber' variable is the layer number that provides
%       the classification scores, i.e., probability.

% -- We tested ADVISE with the 'alexnet', 'vgg16', 'resnet50', and 
%       'xception' pretrained models on ImageNet. Therefore, please assign
%       one of these architectures' names to the 'net' variable.

net = alexnet;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
networkLength = numel(net.Layers);
inputSize = net.Layers(1).InputSize(1:2);
switch networkLength
    case 25
        convLayerNumber = 14;
        reductionLayerNumber = 24;
        classificationLayer = 25;
        networkName = 'Alexnet';
    case 41
        convLayerNumber = 30;
        reductionLayerNumber = 40;
        classificationLayer = 41;
        networkName = 'VGG16';
    case 177
        convLayerNumber = 173;
        reductionLayerNumber = 176;
        classificationLayer = 177;
        networkName = 'ResNet50';
    case 170
        convLayerNumber = 164;
        reductionLayerNumber = 169;
        classificationLayer = 170;
        networkName = 'Xception';
    otherwise
        analyzeNetwork(net)
        error(message("Set proper values based on the architecture"));
end
networkClasses = net.Layers(classificationLayer,1).Classes;
featureLayer = net.Layers(convLayerNumber,1).Name;
reductionLayer = net.Layers(reductionLayerNumber,1).Name;
reductionFunction =[];
dlnet = deep.internal.sdk.dag2dlnetwork(net);

% -- If 'displayFlag' is set to 1, the results of Grad-CAM and LIME
%    will be displayed as well. Setting this entry to 1 increases the 
%    computational complexity.
displayFlag = 1;


%% ADVISE
%-- Select and read image
[file,path] = uigetfile(...
    {'*.jpeg;*.jpg;*.png;*.gif;*.tiff', 'Image file'},...
   'Select an Image', imagesFolder);
if isequal(file,0)
    inputImage = imread('sherlock.jpg');
else
    inputImage = imread([path,file]);
end

[~,~,dim] = size(inputImage);
if dim<3
    inputImage = repmat(inputImage,[1 1 3]);
end

%-- If parallel processing is available, 'parpool' begins using the 
%   available workers on the machine.
if canUseParallelPool
    pool = parpool('local');
    flag = 1;
end

[evaluationMetrics,label] = imADVISE(net,dlnet,inputSize,...
       inputImage,featureLayer,reductionLayer,reductionFunction,...
       networkClasses,displayFlag);

%% Preparing output   
[AVX, idx] = max([evaluationMetrics.XAIValue]);
chNum = num2str(evaluationMetrics(idx).channelNumber);
peakNum = evaluationMetrics(idx).peakCount;

if isa(peakNum,'double')
    message = ['The maximum AVX Value is ', num2str(AVX), ' at #peak ',...
        num2str(peakNum), ', with ', num2str(chNum), ' unit(s).'];
else
    message = ['The maximum AVX Value is ', num2str(AVX), ' with ',...
        peakNum, ' method.'];
end

if flag
    delete(pool);
end

clc
disp(message);
 
clear file path dim flag classificationLayer convLayerNumber dim ...
    displayFlag file networkLength path reductionFunction ...
    reductionLayerNumber AVX idx channelNumber imagesFolder message