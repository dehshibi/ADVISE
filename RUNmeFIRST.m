%% Checking MATLAB version and toolboxes
if isMATLABReleaseOlderThan("R2021a","release")
   disp(['NOTE: This code was developed using MATLAB R2021a. ', ...
   'It might not work properly with the version you have installed.']);  
end

%-- By running the following code, you will be able to see the names of 
%   the various pre-trained networks. If you haven't already installed 
%   'alexnet', 'vgg16', 'resnet50' and 'xception', click install. The 
%   architecture and accompanying weights will be downloaded in this 
%   manner, and you can execute 'imMain.m' to evaluate ADVISE's 
%   functionality.
% For more details, please visit the following links:
% 1. https://www.mathworks.com/help/deeplearning/ref/deepnetworkdesigner-app.html
% 2. https://www.mathworks.com/help/deeplearning/ref/vgg16.html
deepNetworkDesigner()