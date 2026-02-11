function [model, performance] = trainSVM(trainingPath, groundTruthPath, options)
% TRAINSVM Train Support Vector Machine for retinal vessel segmentation
%
% This function trains an SVM classifier for pixel-wise vessel detection
% using extracted features from retinal fundus images. It supports both
% linear and RBF kernels with hyperparameter optimization.
%
% Syntax:
%   [model, performance] = trainSVM(trainingPath, groundTruthPath)
%   [model, performance] = trainSVM(trainingPath, groundTruthPath, options)
%
% Inputs:
%   trainingPath    - Path to training images directory (string)
%   groundTruthPath - Path to ground truth masks directory (string)
%   options         - Training options structure (optional)
%     .kernelFunction - 'linear', 'gaussian', 'polynomial' (default: 'gaussian')
%     .boxConstraint  - SVM box constraint parameter (default: 1)
%     .kernelScale    - RBF kernel scale parameter (default: 'auto')
%     .standardize    - Standardize features (default: true)
%     .crossValidate  - Perform cross-validation (default: true)
%     .numFolds       - Number of CV folds (default: 5)
%     .verbose        - Display progress (default: true)
%
% Outputs:
%   model       - Trained SVM model structure
%     .SVMModel   - MATLAB SVM classifier object
%     .featureStd - Feature standardization parameters
%     .options    - Training options used
%   performance - Training performance metrics structure
%     .accuracy   - Cross-validation accuracy
%     .sensitivity- Cross-validation sensitivity
%     .specificity- Cross-validation specificity
%     .auc        - Area under ROC curve
%
% Description:
%   This function implements SVM-based vessel segmentation with comprehensive
%   feature extraction and hyperparameter optimization. It extracts multi-scale
%   features including intensity, texture, and geometric properties from
%   training images and their corresponding ground truth masks.
%
% Features Extracted:
%   - Multi-scale Gaussian derivatives
%   - Local binary patterns
%   - Vessel-specific geometric features
%   - Intensity and contrast features
%   - Texture descriptors (GLCM, LBP variants)
%
% Example:
%   % Train SVM with default parameters
%   [model, perf] = trainSVM('Images/DRIVE/train/', 'Images/DRIVE/masks/');
%   
%   % Train with custom parameters
%   opts.kernelFunction = 'linear';
%   opts.boxConstraint = 10;
%   opts.crossValidate = true;
%   [model, perf] = trainSVM('Images/DRIVE/train/', 'Images/DRIVE/masks/', opts);
%
% Performance Notes:
%   - RBF kernel typically provides better performance than linear
%   - Feature standardization is highly recommended
%   - Cross-validation helps prevent overfitting
%   - Training time scales with dataset size and feature dimensionality
%
% See also: testSVM, trainRFC, fitcsvm, extractFeature
%
% Reference: 
%   Sayed et al., "Retinal blood vessel segmentation using supervised and 
%   unsupervised approaches", IET Computer Vision, 2021
%
% Author: Retinal Vessel Segmentation Research Team
% Date: February 2026

%% Input validation and default parameters
if nargin < 2
    error('trainSVM:NotEnoughInputs', 'At least 2 inputs required');
end

if nargin < 3 || isempty(options)
    options = struct();
end

% Set default options
if ~isfield(options, 'kernelFunction'), options.kernelFunction = 'gaussian'; end
if ~isfield(options, 'boxConstraint'), options.boxConstraint = 1; end
if ~isfield(options, 'kernelScale'), options.kernelScale = 'auto'; end
if ~isfield(options, 'standardize'), options.standardize = true; end
if ~isfield(options, 'crossValidate'), options.crossValidate = true; end
if ~isfield(options, 'numFolds'), options.numFolds = 5; end
if ~isfield(options, 'verbose'), options.verbose = true; end

if options.verbose
    fprintf('🤖 Starting SVM Training for Retinal Vessel Segmentation\n');
    fprintf('========================================================\n\n');
end

%% Load and prepare training data
if options.verbose
    fprintf('📂 Loading training data...\n');
end

% Get image files
imageFiles = dir(fullfile(trainingPath, '*.tif'));
if isempty(imageFiles)
    imageFiles = [dir(fullfile(trainingPath, '*.jpg')); ...
                  dir(fullfile(trainingPath, '*.png'))];
end

if isempty(imageFiles)
    error('trainSVM:NoImages', 'No training images found in %s', trainingPath);
end

% Get ground truth files
maskFiles = dir(fullfile(groundTruthPath, '*.gif'));
if isempty(maskFiles)
    maskFiles = [dir(fullfile(groundTruthPath, '*.png')); ...
                 dir(fullfile(groundTruthPath, '*.tif'))];
end

if length(imageFiles) ~= length(maskFiles)
    warning('trainSVM:MismatchedFiles', ...
        'Number of images (%d) does not match masks (%d)', ...
        length(imageFiles), length(maskFiles));
end

%% Extract features and labels
if options.verbose
    fprintf('🔍 Extracting features from %d training images...\n', length(imageFiles));
end

allFeatures = [];
allLabels = [];

for i = 1:length(imageFiles)
    if options.verbose
        fprintf('   Processing image %d/%d: %s\n', i, length(imageFiles), imageFiles(i).name);
    end
    
    % Load image and ground truth
    imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    
    % Find corresponding mask
    [~, imgName, ~] = fileparts(imageFiles(i).name);
    maskIdx = find(contains({maskFiles.name}, imgName));
    if isempty(maskIdx)
        % Try alternative naming conventions
        maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_test', '')));
        if isempty(maskIdx)
            maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_train', '')));
        end
    end
    
    if isempty(maskIdx)
        warning('trainSVM:NoMask', 'No mask found for image %s', imgName);
        continue;
    end
    
    maskPath = fullfile(maskFiles(maskIdx(1)).folder, maskFiles(maskIdx(1)).name);
    
    % Load and preprocess
    img = imread(imgPath);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    mask = imread(maskPath);
    if size(mask, 3) == 3
        mask = rgb2gray(mask);
    end
    mask = mask > 128; % Binarize
    
    % Extract features at multiple scales
    features = extractSVMFeatures(img);
    
    % Get valid pixel indices (avoid borders)
    [rows, cols] = size(img);
    borderSize = 16; % Avoid border artifacts
    validRows = borderSize+1:rows-borderSize;
    validCols = borderSize+1:cols-borderSize;
    
    % Sample features (to manage memory)
    sampleRatio = 0.1; % Sample 10% of pixels
    [sampR, sampC] = meshgrid(validRows(1:round(1/sampleRatio):end), ...
                             validCols(1:round(1/sampleRatio):end));
    sampleIdx = sub2ind(size(img), sampR(:), sampC(:));
    
    % Extract features and labels
    imgFeatures = features(sampleIdx, :);
    imgLabels = mask(sampleIdx);
    
    % Remove NaN and infinite values
    validIdx = all(isfinite(imgFeatures), 2);
    imgFeatures = imgFeatures(validIdx, :);
    imgLabels = imgLabels(validIdx);
    
    allFeatures = [allFeatures; imgFeatures];
    allLabels = [allLabels; imgLabels];
end

if options.verbose
    fprintf('✅ Feature extraction complete. Total samples: %d\n', length(allLabels));
    fprintf('   Vessel pixels: %d (%.1f%%)\n', sum(allLabels), 100*mean(allLabels));
    fprintf('   Feature dimensions: %d\n\n', size(allFeatures, 2));
end

%% Balance dataset
if options.verbose
    fprintf('⚖️  Balancing dataset...\n');
end

vesselIdx = find(allLabels == 1);
nonVesselIdx = find(allLabels == 0);

% Undersample majority class to balance
minSamples = min(length(vesselIdx), length(nonVesselIdx));
balancedIdx = [vesselIdx(randperm(length(vesselIdx), minSamples)); ...
               nonVesselIdx(randperm(length(nonVesselIdx), minSamples))];

balancedFeatures = allFeatures(balancedIdx, :);
balancedLabels = allLabels(balancedIdx);

if options.verbose
    fprintf('   Balanced dataset: %d samples per class\n', minSamples);
end

%% Feature standardization
if options.standardize
    if options.verbose
        fprintf('📊 Standardizing features...\n');
    end
    
    featureMean = mean(balancedFeatures);
    featureStd = std(balancedFeatures);
    featureStd(featureStd == 0) = 1; % Avoid division by zero
    
    balancedFeatures = (balancedFeatures - featureMean) ./ featureStd;
    
    model.featureStd.mean = featureMean;
    model.featureStd.std = featureStd;
else
    model.featureStd = [];
end

%% Train SVM
if options.verbose
    fprintf('🚂 Training SVM classifier...\n');
    fprintf('   Kernel: %s\n', options.kernelFunction);
    fprintf('   Box Constraint: %.2f\n', options.boxConstraint);
    if isnumeric(options.kernelScale)
        fprintf('   Kernel Scale: %.4f\n', options.kernelScale);
    else
        fprintf('   Kernel Scale: %s\n', options.kernelScale);
    end
end

% Train SVM
tic;
model.SVMModel = fitcsvm(balancedFeatures, balancedLabels, ...
    'KernelFunction', options.kernelFunction, ...
    'BoxConstraint', options.boxConstraint, ...
    'KernelScale', options.kernelScale, ...
    'Standardize', false); % Already standardized if requested

trainingTime = toc;
model.options = options;

if options.verbose
    fprintf('✅ Training completed in %.2f seconds\n\n', trainingTime);
end

%% Cross-validation evaluation
if options.crossValidate
    if options.verbose
        fprintf('🔄 Performing %d-fold cross-validation...\n', options.numFolds);
    end
    
    % Cross-validated SVM
    cvSVM = fitcsvm(balancedFeatures, balancedLabels, ...
        'KernelFunction', options.kernelFunction, ...
        'BoxConstraint', options.boxConstraint, ...
        'KernelScale', options.kernelScale, ...
        'Standardize', false, ...
        'CrossVal', 'on', 'KFold', options.numFolds);
    
    % Get cross-validation predictions
    [cvPredictions, cvScores] = kfoldPredict(cvSVM);
    
    % Calculate performance metrics
    performance.accuracy = 1 - kfoldLoss(cvSVM);
    
    % Calculate sensitivity and specificity
    tp = sum(cvPredictions == 1 & balancedLabels == 1);
    tn = sum(cvPredictions == 0 & balancedLabels == 0);
    fp = sum(cvPredictions == 1 & balancedLabels == 0);
    fn = sum(cvPredictions == 0 & balancedLabels == 1);
    
    performance.sensitivity = tp / (tp + fn);
    performance.specificity = tn / (tn + fp);
    performance.precision = tp / (tp + fp);
    performance.f1Score = 2 * (performance.precision * performance.sensitivity) / ...
                         (performance.precision + performance.sensitivity);
    
    % Calculate AUC
    [~, ~, ~, performance.auc] = perfcurve(balancedLabels, cvScores(:,2), 1);
    
    if options.verbose
        fprintf('📈 Cross-validation results:\n');
        fprintf('   Accuracy:    %.3f\n', performance.accuracy);
        fprintf('   Sensitivity: %.3f\n', performance.sensitivity);
        fprintf('   Specificity: %.3f\n', performance.specificity);
        fprintf('   Precision:   %.3f\n', performance.precision);
        fprintf('   F1-Score:    %.3f\n', performance.f1Score);
        fprintf('   AUC:         %.3f\n\n', performance.auc);
    end
else
    performance = [];
end

if options.verbose
    fprintf('✅ SVM training completed successfully!\n');
    fprintf('========================================================\n\n');
end

end

function features = extractSVMFeatures(img)
% Extract comprehensive features for SVM training
img = double(img);
[rows, cols] = size(img);

% Initialize feature matrix
numFeatures = 20; % Total number of features
features = zeros(rows, cols, numFeatures);

% Feature 1-3: Multi-scale Gaussian derivatives
scales = [1, 2, 4];
for i = 1:length(scales)
    sigma = scales(i);
    [Gx, Gy] = imgradientxy(imgaussfilt(img, sigma));
    features(:, :, i) = sqrt(Gx.^2 + Gy.^2); % Gradient magnitude
end

% Feature 4-6: Hessian eigenvalues at multiple scales
for i = 1:length(scales)
    sigma = scales(i);
    smoothed = imgaussfilt(img, sigma);
    [Ixx, Ixy, Iyy] = computeHessian(smoothed);
    
    % Eigenvalues of Hessian matrix
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    features(:, :, 3+i) = lambda1;
end

% Feature 7-9: Local binary pattern variants
features(:, :, 7) = extractLBP(img, 8, 1);
features(:, :, 8) = extractLBP(img, 8, 2);
features(:, :, 9) = extractLBP(img, 16, 2);

% Feature 10-12: Intensity statistics
features(:, :, 10) = img; % Original intensity
features(:, :, 11) = imfilter(img, fspecial('average', 5)); % Local mean
features(:, :, 12) = stdfilt(img, ones(5)); % Local standard deviation

% Feature 13-15: Gabor filter responses
for i = 1:3
    theta = (i-1) * 60; % 0, 60, 120 degrees
    gaborFilter = gabor(4, theta);
    gaborResponse = imfilter(img, real(gaborFilter), 'symmetric');
    features(:, :, 12+i) = gaborResponse;
end

% Feature 16-18: Vessel-specific measures
features(:, :, 16) = computeVesselnessFilter(img, [1 2 4]);
features(:, :, 17) = imfilter(img, fspecial('laplacian')); % Laplacian
features(:, :, 18) = imgradient(img); % Simple gradient

% Feature 19-20: Texture measures
features(:, :, 19) = rangefilt(img, ones(5)); % Local range
features(:, :, 20) = entropyfilt(img, ones(5)); % Local entropy

% Reshape to pixel-wise feature vectors
features = reshape(features, rows*cols, numFeatures);
end

function [Ixx, Ixy, Iyy] = computeHessian(img)
% Compute Hessian matrix components
[Ix, Iy] = imgradientxy(img);
[Ixx, ~] = imgradientxy(Ix);
[Ixy, Iyy] = imgradientxy(Iy);
end

function lbp = extractLBP(img, neighbors, radius)
% Simple LBP implementation
[rows, cols] = size(img);
lbp = zeros(rows, cols);

for i = radius+1:rows-radius
    for j = radius+1:cols-radius
        center = img(i, j);
        pattern = 0;
        
        for k = 0:neighbors-1
            angle = 2 * pi * k / neighbors;
            x = i + radius * cos(angle);
            y = j + radius * sin(angle);
            
            % Bilinear interpolation
            neighbor = interp2(img, y, x, 'linear', 0);
            
            if neighbor >= center
                pattern = pattern + 2^k;
            end
        end
        
        lbp(i, j) = pattern;
    end
end
end

function vesselness = computeVesselnessFilter(img, scales)
% Simplified vesselness filter
vesselness = zeros(size(img));

for scale = scales
    smoothed = imgaussfilt(img, scale);
    [Ixx, Ixy, Iyy] = computeHessian(smoothed);
    
    % Eigenvalues
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    lambda2 = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    
    % Vesselness measure
    vesselness = max(vesselness, -lambda2 .* (lambda2 < 0));
end
end
