function [model, performance] = trainAdaBoost(trainingPath, groundTruthPath, options)
% TRAINADABOOST Train AdaBoost classifier for retinal vessel segmentation
%
% This function trains an AdaBoost (Adaptive Boosting) ensemble classifier
% for pixel-wise vessel detection using extracted features from retinal
% fundus images. AdaBoost combines multiple weak learners to create a
% strong classifier with improved generalization.
%
% Syntax:
%   [model, performance] = trainAdaBoost(trainingPath, groundTruthPath)
%   [model, performance] = trainAdaBoost(trainingPath, groundTruthPath, options)
%
% Inputs:
%   trainingPath    - Path to training images directory (string)
%   groundTruthPath - Path to ground truth masks directory (string)
%   options         - Training options structure (optional)
%     .method         - AdaBoost method ('AdaBoostM1', 'AdaBoostM2', 'GentleBoost', 'LogitBoost')
%     .numLearners    - Number of weak learners (default: 100)
%     .learnerType    - Weak learner type ('Tree', 'Discriminant') (default: 'Tree')
%     .maxNumSplits   - Maximum splits per tree (default: 10)
%     .learningRate   - Learning rate for boosting (default: 0.1)
%     .crossValidate  - Perform cross-validation (default: true)
%     .numFolds       - Number of CV folds (default: 5)
%     .verbose        - Display progress (default: true)
%
% Outputs:
%   model       - Trained AdaBoost model structure
%     .AdaModel   - MATLAB ensemble classifier object
%     .featureStd - Feature standardization parameters
%     .options    - Training options used
%   performance - Training performance metrics structure
%     .accuracy   - Cross-validation accuracy
%     .sensitivity- Cross-validation sensitivity
%     .specificity- Cross-validation specificity
%     .auc        - Area under ROC curve
%     .oobError   - Out-of-bag error
%
% Description:
%   AdaBoost is particularly effective for retinal vessel segmentation due
%   to its ability to focus on difficult-to-classify vessel pixels and
%   handle class imbalance naturally. This implementation supports various
%   boosting algorithms and weak learner types.
%
% Features Used:
%   - Multi-scale intensity gradients
%   - Hessian-based vessel filters
%   - Local binary patterns
%   - Gabor filter responses
%   - Morphological features
%   - Texture descriptors
%
% Example:
%   % Train AdaBoost with default parameters
%   [model, perf] = trainAdaBoost('Images/DRIVE/train/', 'Images/DRIVE/masks/');
%   
%   % Train with custom parameters
%   opts.method = 'GentleBoost';
%   opts.numLearners = 200;
%   opts.maxNumSplits = 20;
%   [model, perf] = trainAdaBoost('Images/DRIVE/train/', 'Images/DRIVE/masks/', opts);
%
% Performance Notes:
%   - GentleBoost typically provides better performance than AdaBoostM1
%   - More learners improve accuracy but increase training time
%   - Tree-based weak learners are recommended for vessel segmentation
%
% See also: testAdaBoost, trainRFC, trainSVM, fitcensemble
%
% Reference: 
%   Sayed et al., "Retinal blood vessel segmentation using supervised and 
%   unsupervised approaches", IET Computer Vision, 2021
%
% Author: Retinal Vessel Segmentation Research Team
% Date: February 2026

%% Input validation and default parameters
if nargin < 2
    error('trainAdaBoost:NotEnoughInputs', 'At least 2 inputs required');
end

if nargin < 3 || isempty(options)
    options = struct();
end

% Set default options
if ~isfield(options, 'method'), options.method = 'GentleBoost'; end
if ~isfield(options, 'numLearners'), options.numLearners = 100; end
if ~isfield(options, 'learnerType'), options.learnerType = 'Tree'; end
if ~isfield(options, 'maxNumSplits'), options.maxNumSplits = 10; end
if ~isfield(options, 'learningRate'), options.learningRate = 0.1; end
if ~isfield(options, 'crossValidate'), options.crossValidate = true; end
if ~isfield(options, 'numFolds'), options.numFolds = 5; end
if ~isfield(options, 'verbose'), options.verbose = true; end

if options.verbose
    fprintf('🚀 Starting AdaBoost Training for Retinal Vessel Segmentation\n');
    fprintf('===========================================================\n\n');
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
    error('trainAdaBoost:NoImages', 'No training images found in %s', trainingPath);
end

% Get ground truth files
maskFiles = dir(fullfile(groundTruthPath, '*.gif'));
if isempty(maskFiles)
    maskFiles = [dir(fullfile(groundTruthPath, '*.png')); ...
                 dir(fullfile(groundTruthPath, '*.tif'))];
end

if length(imageFiles) ~= length(maskFiles)
    warning('trainAdaBoost:MismatchedFiles', ...
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
        maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_test', '')));
        if isempty(maskIdx)
            maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_train', '')));
        end
    end
    
    if isempty(maskIdx)
        warning('trainAdaBoost:NoMask', 'No mask found for image %s', imgName);
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
    
    % Extract comprehensive features for AdaBoost
    features = extractAdaBoostFeatures(img);
    
    % Get valid pixel indices (avoid borders)
    [rows, cols] = size(img);
    borderSize = 20; % Larger border for AdaBoost features
    validRows = borderSize+1:rows-borderSize;
    validCols = borderSize+1:cols-borderSize;
    
    % Strategic sampling for AdaBoost (focus on vessel boundaries)
    sampleRatio = 0.05; % 5% sampling
    [sampR, sampC] = meshgrid(validRows(1:round(1/sampleRatio):end), ...
                             validCols(1:round(1/sampleRatio):end));
    sampleIdx = sub2ind(size(img), sampR(:), sampC(:));
    
    % Add additional samples near vessel boundaries
    vesselBoundary = edge(mask, 'canny');
    [boundaryR, boundaryC] = find(vesselBoundary);
    if ~isempty(boundaryR)
        boundaryIdx = sub2ind(size(img), boundaryR, boundaryC);
        % Sample boundary pixels
        nBoundarySamples = min(length(boundaryIdx), round(length(sampleIdx) * 0.3));
        boundarySelection = boundaryIdx(randperm(length(boundaryIdx), nBoundarySamples));
        sampleIdx = [sampleIdx; boundarySelection];
    end
    
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

%% Handle class imbalance (AdaBoost can handle some imbalance naturally)
vesselIdx = find(allLabels == 1);
nonVesselIdx = find(allLabels == 0);

% Moderate balancing (don't oversample too much for AdaBoost)
vesselRatio = length(vesselIdx) / length(allLabels);
if vesselRatio < 0.2  % If vessel pixels < 20%, balance to 30%
    targetVesselSamples = round(length(nonVesselIdx) * 0.3 / 0.7);
    if targetVesselSamples > length(vesselIdx)
        % Oversample vessel pixels
        oversampleIdx = vesselIdx(randi(length(vesselIdx), targetVesselSamples - length(vesselIdx), 1));
        balancedIdx = [vesselIdx; nonVesselIdx; oversampleIdx];
    else
        % Use all vessel pixels
        balancedIdx = [vesselIdx; nonVesselIdx];
    end
else
    balancedIdx = 1:length(allLabels);
end

balancedFeatures = allFeatures(balancedIdx, :);
balancedLabels = allLabels(balancedIdx);

if options.verbose
    fprintf('⚖️  Dataset balancing complete.\n');
    fprintf('   Final samples: %d\n', length(balancedLabels));
    fprintf('   Vessel ratio: %.1f%%\n\n', 100*mean(balancedLabels));
end

%% Feature standardization (optional for tree-based learners)
featureMean = mean(balancedFeatures);
featureStd = std(balancedFeatures);
featureStd(featureStd == 0) = 1; % Avoid division by zero

% Light standardization (preserve feature interpretability for trees)
balancedFeatures = (balancedFeatures - featureMean) ./ (featureStd + 1e-6);

model.featureStd.mean = featureMean;
model.featureStd.std = featureStd;

%% Configure weak learner template
if strcmpi(options.learnerType, 'Tree')
    weakLearner = templateTree('MaxNumSplits', options.maxNumSplits, ...
                              'SplitCriterion', 'gdi');
elseif strcmpi(options.learnerType, 'Discriminant')
    weakLearner = templateDiscriminant();
else
    error('trainAdaBoost:InvalidLearner', 'Unsupported learner type: %s', options.learnerType);
end

%% Train AdaBoost
if options.verbose
    fprintf('🚂 Training AdaBoost classifier...\n');
    fprintf('   Method: %s\n', options.method);
    fprintf('   Number of learners: %d\n', options.numLearners);
    fprintf('   Weak learner: %s\n', options.learnerType);
    if strcmpi(options.learnerType, 'Tree')
        fprintf('   Max splits per tree: %d\n', options.maxNumSplits);
    end
    fprintf('   Learning rate: %.3f\n', options.learningRate);
end

% Train AdaBoost ensemble
tic;
model.AdaModel = fitcensemble(balancedFeatures, balancedLabels, ...
    'Method', options.method, ...
    'NumLearningCycles', options.numLearners, ...
    'Learners', weakLearner, ...
    'LearnRate', options.learningRate);

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
    
    % Cross-validated AdaBoost
    cvAdaBoost = fitcensemble(balancedFeatures, balancedLabels, ...
        'Method', options.method, ...
        'NumLearningCycles', options.numLearners, ...
        'Learners', weakLearner, ...
        'LearnRate', options.learningRate, ...
        'CrossVal', 'on', 'KFold', options.numFolds);
    
    % Get cross-validation predictions
    [cvPredictions, cvScores] = kfoldPredict(cvAdaBoost);
    
    % Calculate performance metrics
    performance.accuracy = 1 - kfoldLoss(cvAdaBoost);
    
    % Calculate detailed metrics
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
    
    % Out-of-bag error
    performance.oobError = oobLoss(model.AdaModel);
    
    % Feature importance (for tree-based learners)
    if strcmpi(options.learnerType, 'Tree')
        performance.featureImportance = predictorImportance(model.AdaModel);
    end
    
    if options.verbose
        fprintf('📈 Cross-validation results:\n');
        fprintf('   Accuracy:    %.3f\n', performance.accuracy);
        fprintf('   Sensitivity: %.3f\n', performance.sensitivity);
        fprintf('   Specificity: %.3f\n', performance.specificity);
        fprintf('   Precision:   %.3f\n', performance.precision);
        fprintf('   F1-Score:    %.3f\n', performance.f1Score);
        fprintf('   AUC:         %.3f\n', performance.auc);
        fprintf('   OOB Error:   %.3f\n', performance.oobError);
        
        if strcmpi(options.learnerType, 'Tree')
            fprintf('   Top 5 important features: ');
            [~, sortIdx] = sort(performance.featureImportance, 'descend');
            fprintf('%d ', sortIdx(1:min(5, length(sortIdx))));
            fprintf('\n');
        end
        fprintf('\n');
    end
else
    performance = [];
end

if options.verbose
    fprintf('✅ AdaBoost training completed successfully!\n');
    fprintf('===========================================================\n\n');
end

end

function features = extractAdaBoostFeatures(img)
% Extract comprehensive features optimized for AdaBoost learning
img = double(img);
[rows, cols] = size(img);

% Initialize feature matrix
numFeatures = 25; % Extended feature set for AdaBoost
features = zeros(rows, cols, numFeatures);

% Feature 1-4: Multi-scale Gaussian derivatives
scales = [0.5, 1, 2, 4];
for i = 1:length(scales)
    sigma = scales(i);
    [Gx, Gy] = imgradientxy(imgaussfilt(img, sigma));
    features(:, :, i) = sqrt(Gx.^2 + Gy.^2); % Gradient magnitude
end

% Feature 5-8: Hessian-based vessel measures
for i = 1:length(scales)
    sigma = scales(i);
    smoothed = imgaussfilt(img, sigma);
    [Ixx, Ixy, Iyy] = computeHessianAdaBoost(smoothed);
    
    % Vesselness measure (Frangi-like)
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    lambda2 = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    vesselness = -lambda2 .* (lambda2 < 0 & abs(lambda1) < abs(lambda2));
    features(:, :, 4+i) = vesselness;
end

% Feature 9-12: Directional derivatives
directions = [0, 45, 90, 135]; % degrees
for i = 1:length(directions)
    angle = directions(i) * pi / 180;
    kernel = [-sin(angle), 0, sin(angle); 
              -cos(angle), 0, cos(angle);
              -sin(angle), 0, sin(angle)] / 3;
    features(:, :, 8+i) = imfilter(img, kernel, 'symmetric');
end

% Feature 13-16: Local binary patterns (multiple radii)
features(:, :, 13) = extractLBPAdaBoost(img, 8, 1);
features(:, :, 14) = extractLBPAdaBoost(img, 8, 2);
features(:, :, 15) = extractLBPAdaBoost(img, 16, 2);
features(:, :, 16) = extractLBPAdaBoost(img, 16, 3);

% Feature 17-19: Gabor filters
gabor_angles = [0, 45, 90];
for i = 1:length(gabor_angles)
    gaborFilter = gabor(2, gabor_angles(i));
    features(:, :, 16+i) = abs(imfilter(img, real(gaborFilter), 'symmetric'));
end

% Feature 20-22: Morphological operations
se1 = strel('disk', 1);
se2 = strel('disk', 2);
features(:, :, 20) = imopen(img, se1); % Opening
features(:, :, 21) = imclose(img, se1); % Closing
features(:, :, 22) = img - imopen(img, se2); % Top-hat

% Feature 23-25: Texture measures
features(:, :, 23) = rangefilt(img, ones(3)); % Local range
features(:, :, 24) = stdfilt(img, ones(5)); % Local std
features(:, :, 25) = entropyfilt(img, ones(5)); % Local entropy

% Reshape to pixel-wise feature vectors
features = reshape(features, rows*cols, numFeatures);

% Remove any remaining NaN or Inf values
features(~isfinite(features)) = 0;
end

function [Ixx, Ixy, Iyy] = computeHessianAdaBoost(img)
% Compute Hessian matrix components
[Ix, Iy] = imgradientxy(img);
[Ixx, ~] = imgradientxy(Ix);
[Ixy, Iyy] = imgradientxy(Iy);
end

function lbp = extractLBPAdaBoost(img, neighbors, radius)
% Local Binary Pattern implementation optimized for vessel detection
[rows, cols] = size(img);
lbp = zeros(rows, cols);

% Precompute neighbor coordinates
angles = 2 * pi * (0:neighbors-1) / neighbors;
neighbor_x = radius * cos(angles);
neighbor_y = radius * sin(angles);

for i = radius+1:rows-radius
    for j = radius+1:cols-radius
        center = img(i, j);
        pattern = 0;
        
        for k = 1:neighbors
            % Bilinear interpolation for sub-pixel accuracy
            x = i + neighbor_y(k);
            y = j + neighbor_x(k);
            
            x1 = floor(x); x2 = x1 + 1;
            y1 = floor(y); y2 = y1 + 1;
            
            if x2 <= rows && y2 <= cols && x1 >= 1 && y1 >= 1
                f11 = img(x1, y1); f12 = img(x1, y2);
                f21 = img(x2, y1); f22 = img(x2, y2);
                
                neighbor_val = f11*(x2-x)*(y2-y) + f21*(x-x1)*(y2-y) + ...
                              f12*(x2-x)*(y-y1) + f22*(x-x1)*(y-y1);
            else
                neighbor_val = center; % Boundary handling
            end
            
            if neighbor_val >= center
                pattern = pattern + 2^(k-1);
            end
        end
        
        lbp(i, j) = pattern;
    end
end
end
