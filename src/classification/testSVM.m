function [predictions, scores, performance] = testSVM(model, testPath, groundTruthPath)
% TESTSVM Test trained SVM model on retinal vessel segmentation
%
% This function applies a trained SVM model to test images for retinal
% vessel segmentation and evaluates performance against ground truth masks.
%
% Syntax:
%   predictions = testSVM(model, testPath)
%   [predictions, scores] = testSVM(model, testPath)
%   [predictions, scores, performance] = testSVM(model, testPath, groundTruthPath)
%
% Inputs:
%   model           - Trained SVM model from trainSVM function
%   testPath        - Path to test images directory (string)
%   groundTruthPath - Path to ground truth masks directory (optional, string)
%
% Outputs:
%   predictions - Cell array of binary segmentation masks
%   scores      - Cell array of probability/confidence maps  
%   performance - Performance metrics structure (if ground truth provided)
%     .accuracy   - Overall pixel accuracy
%     .sensitivity- Sensitivity (true positive rate)
%     .specificity- Specificity (true negative rate)
%     .precision  - Precision (positive predictive value)
%     .f1Score    - F1-score (harmonic mean of precision and sensitivity)
%     .auc        - Area under ROC curve
%     .perImage   - Per-image performance metrics
%
% Description:
%   This function performs comprehensive testing of SVM-based vessel
%   segmentation including feature extraction, classification, and
%   post-processing. It provides both pixel-wise predictions and
%   confidence scores for each test image.
%
% Post-processing Steps:
%   - Morphological opening to remove noise
%   - Connected component analysis
%   - Size-based filtering of small components
%   - Optional hole filling for vessel continuity
%
% Example:
%   % Test without performance evaluation
%   predictions = testSVM(trainedModel, 'Images/DRIVE/test/');
%   
%   % Test with performance evaluation
%   [pred, scores, perf] = testSVM(trainedModel, ...
%       'Images/DRIVE/test/', 'Images/DRIVE/masks/');
%   
%   % Display results
%   figure; imshow(predictions{1});
%   fprintf('Overall accuracy: %.3f\n', perf.accuracy);
%
% Performance Notes:
%   - Processing time depends on image size and feature complexity
%   - Memory usage scales with image dimensions
%   - Post-processing significantly improves visual quality
%
% See also: trainSVM, extractFeature, accuracy_tesst
%
% Reference: 
%   Sayed et al., "Retinal blood vessel segmentation using supervised and 
%   unsupervised approaches", IET Computer Vision, 2021
%
% Author: Retinal Vessel Segmentation Research Team
% Date: February 2026

%% Input validation
if nargin < 2
    error('testSVM:NotEnoughInputs', 'At least 2 inputs required');
end

evaluatePerformance = (nargin >= 3) && ~isempty(groundTruthPath);

fprintf('🧪 Testing SVM Model on Retinal Images\n');
fprintf('=====================================\n\n');

%% Load test images
fprintf('📂 Loading test images...\n');

imageFiles = dir(fullfile(testPath, '*.tif'));
if isempty(imageFiles)
    imageFiles = [dir(fullfile(testPath, '*.jpg')); ...
                  dir(fullfile(testPath, '*.png'))];
end

if isempty(imageFiles)
    error('testSVM:NoImages', 'No test images found in %s', testPath);
end

fprintf('   Found %d test images\n\n', length(imageFiles));

%% Load ground truth if provided
if evaluatePerformance
    fprintf('📊 Loading ground truth masks...\n');
    
    maskFiles = dir(fullfile(groundTruthPath, '*.gif'));
    if isempty(maskFiles)
        maskFiles = [dir(fullfile(groundTruthPath, '*.png')); ...
                     dir(fullfile(groundTruthPath, '*.tif'))];
    end
    
    if length(imageFiles) ~= length(maskFiles)
        warning('testSVM:MismatchedFiles', ...
            'Number of images (%d) does not match masks (%d)', ...
            length(imageFiles), length(maskFiles));
    end
    
    fprintf('   Found %d ground truth masks\n\n', length(maskFiles));
end

%% Initialize outputs
predictions = cell(length(imageFiles), 1);
scores = cell(length(imageFiles), 1);

if evaluatePerformance
    performance.perImage.accuracy = zeros(length(imageFiles), 1);
    performance.perImage.sensitivity = zeros(length(imageFiles), 1);
    performance.perImage.specificity = zeros(length(imageFiles), 1);
    performance.perImage.precision = zeros(length(imageFiles), 1);
    performance.perImage.f1Score = zeros(length(imageFiles), 1);
    performance.perImage.auc = zeros(length(imageFiles), 1);
    
    allPredictions = [];
    allGroundTruth = [];
    allScores = [];
end

%% Process each test image
fprintf('🔍 Processing test images...\n');

for i = 1:length(imageFiles)
    fprintf('   Image %d/%d: %s\n', i, length(imageFiles), imageFiles(i).name);
    
    %% Load and preprocess image
    imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    img = imread(imgPath);
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    originalSize = size(img);
    
    %% Extract features
    features = extractSVMFeatures(img);
    
    % Apply feature standardization if used during training
    if ~isempty(model.featureStd)
        features = (features - model.featureStd.mean) ./ model.featureStd.std;
    end
    
    % Remove invalid features
    validIdx = all(isfinite(features), 2);
    
    %% Classify pixels
    pixelPredictions = zeros(size(features, 1), 1);
    pixelScores = zeros(size(features, 1), 2);
    
    if any(validIdx)
        try
            [pixelPredictions(validIdx), pixelScores(validIdx, :)] = ...
                predict(model.SVMModel, features(validIdx, :));
        catch ME
            warning('testSVM:PredictionError', ...
                'Error predicting image %d: %s', i, ME.message);
            continue;
        end
    end
    
    %% Reshape predictions back to image format
    predictionMap = reshape(pixelPredictions, originalSize);
    scoreMap = reshape(pixelScores(:, 2), originalSize); % Probability of vessel class
    
    %% Post-processing
    processedPrediction = postProcessSegmentation(predictionMap);
    
    %% Store results
    predictions{i} = processedPrediction;
    scores{i} = scoreMap;
    
    %% Evaluate performance if ground truth available
    if evaluatePerformance
        % Find corresponding ground truth
        [~, imgName, ~] = fileparts(imageFiles(i).name);
        maskIdx = find(contains({maskFiles.name}, imgName));
        
        if isempty(maskIdx)
            % Try alternative naming conventions
            maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_test', '')));
            if isempty(maskIdx)
                maskIdx = find(contains({maskFiles.name}, strrep(imgName, '_train', '')));
            end
        end
        
        if ~isempty(maskIdx)
            maskPath = fullfile(maskFiles(maskIdx(1)).folder, maskFiles(maskIdx(1)).name);
            groundTruth = imread(maskPath);
            
            if size(groundTruth, 3) == 3
                groundTruth = rgb2gray(groundTruth);
            end
            groundTruth = groundTruth > 128; % Binarize
            
            % Resize ground truth if necessary
            if ~isequal(size(groundTruth), size(processedPrediction))
                groundTruth = imresize(groundTruth, size(processedPrediction), 'nearest');
            end
            
            % Calculate per-image metrics
            metrics = calculateMetrics(processedPrediction, groundTruth, scoreMap);
            
            performance.perImage.accuracy(i) = metrics.accuracy;
            performance.perImage.sensitivity(i) = metrics.sensitivity;
            performance.perImage.specificity(i) = metrics.specificity;
            performance.perImage.precision(i) = metrics.precision;
            performance.perImage.f1Score(i) = metrics.f1Score;
            performance.perImage.auc(i) = metrics.auc;
            
            % Accumulate for overall metrics
            allPredictions = [allPredictions; processedPrediction(:)];
            allGroundTruth = [allGroundTruth; groundTruth(:)];
            allScores = [allScores; scoreMap(:)];
        else
            warning('testSVM:NoGroundTruth', ...
                'No ground truth found for image %s', imgName);
        end
    end
end

%% Calculate overall performance metrics
if evaluatePerformance && ~isempty(allPredictions)
    fprintf('\n📈 Calculating overall performance metrics...\n');
    
    overallMetrics = calculateMetrics(allPredictions, allGroundTruth, allScores);
    
    performance.accuracy = overallMetrics.accuracy;
    performance.sensitivity = overallMetrics.sensitivity;
    performance.specificity = overallMetrics.specificity;
    performance.precision = overallMetrics.precision;
    performance.f1Score = overallMetrics.f1Score;
    performance.auc = overallMetrics.auc;
    
    % Calculate mean per-image metrics
    performance.meanPerImage.accuracy = mean(performance.perImage.accuracy);
    performance.meanPerImage.sensitivity = mean(performance.perImage.sensitivity);
    performance.meanPerImage.specificity = mean(performance.perImage.specificity);
    performance.meanPerImage.precision = mean(performance.perImage.precision);
    performance.meanPerImage.f1Score = mean(performance.perImage.f1Score);
    performance.meanPerImage.auc = mean(performance.perImage.auc);
    
    % Display results
    fprintf('\n📊 Overall Performance Results:\n');
    fprintf('   Accuracy:    %.3f\n', performance.accuracy);
    fprintf('   Sensitivity: %.3f\n', performance.sensitivity);
    fprintf('   Specificity: %.3f\n', performance.specificity);
    fprintf('   Precision:   %.3f\n', performance.precision);
    fprintf('   F1-Score:    %.3f\n', performance.f1Score);
    fprintf('   AUC:         %.3f\n', performance.auc);
    
    fprintf('\n📊 Mean Per-Image Performance:\n');
    fprintf('   Accuracy:    %.3f ± %.3f\n', ...
        performance.meanPerImage.accuracy, std(performance.perImage.accuracy));
    fprintf('   Sensitivity: %.3f ± %.3f\n', ...
        performance.meanPerImage.sensitivity, std(performance.perImage.sensitivity));
    fprintf('   Specificity: %.3f ± %.3f\n', ...
        performance.meanPerImage.specificity, std(performance.perImage.specificity));
    fprintf('   F1-Score:    %.3f ± %.3f\n', ...
        performance.meanPerImage.f1Score, std(performance.perImage.f1Score));
    fprintf('   AUC:         %.3f ± %.3f\n', ...
        performance.meanPerImage.auc, std(performance.perImage.auc));
end

fprintf('\n✅ SVM testing completed successfully!\n');
fprintf('=====================================\n\n');

end

function processedPrediction = postProcessSegmentation(prediction)
% Post-process binary segmentation mask
    
% Convert to logical
prediction = logical(prediction);

% Morphological opening to remove small noise
se1 = strel('disk', 1);
cleaned = imopen(prediction, se1);

% Connected component analysis
cc = bwconncomp(cleaned);
numPixels = cellfun(@numel, cc.PixelIdxList);

% Remove very small components
minComponentSize = 20; % Minimum vessel segment size
largeComponents = numPixels >= minComponentSize;
processedPrediction = false(size(prediction));

for i = find(largeComponents)
    processedPrediction(cc.PixelIdxList{i}) = true;
end

% Optional: Fill small holes in vessels
% processedPrediction = imfill(processedPrediction, 'holes');

end

function metrics = calculateMetrics(prediction, groundTruth, scores)
% Calculate comprehensive performance metrics

% Convert to logical
prediction = logical(prediction(:));
groundTruth = logical(groundTruth(:));
scores = double(scores(:));

% Confusion matrix components
tp = sum(prediction & groundTruth);
tn = sum(~prediction & ~groundTruth);
fp = sum(prediction & ~groundTruth);
fn = sum(~prediction & groundTruth);

% Basic metrics
metrics.accuracy = (tp + tn) / (tp + tn + fp + fn);
metrics.sensitivity = tp / (tp + fn);
metrics.specificity = tn / (tn + fp);
metrics.precision = tp / (tp + fp);

if metrics.precision + metrics.sensitivity > 0
    metrics.f1Score = 2 * (metrics.precision * metrics.sensitivity) / ...
                     (metrics.precision + metrics.sensitivity);
else
    metrics.f1Score = 0;
end

% AUC calculation
if length(unique(scores)) > 1
    try
        [~, ~, ~, metrics.auc] = perfcurve(groundTruth, scores, true);
    catch
        metrics.auc = 0.5; % Default AUC for constant predictions
    end
else
    metrics.auc = 0.5;
end

% Handle NaN values
if isnan(metrics.sensitivity), metrics.sensitivity = 0; end
if isnan(metrics.specificity), metrics.specificity = 0; end
if isnan(metrics.precision), metrics.precision = 0; end
if isnan(metrics.f1Score), metrics.f1Score = 0; end
if isnan(metrics.auc), metrics.auc = 0.5; end

end

function features = extractSVMFeatures(img)
% Extract the same features used in training (must match trainSVM)
img = double(img);
[rows, cols] = size(img);

% Initialize feature matrix
numFeatures = 20; % Must match training
features = zeros(rows, cols, numFeatures);

% Feature 1-3: Multi-scale Gaussian derivatives
scales = [1, 2, 4];
for i = 1:length(scales)
    sigma = scales(i);
    [Gx, Gy] = imgradientxy(imgaussfilt(img, sigma));
    features(:, :, i) = sqrt(Gx.^2 + Gy.^2);
end

% Feature 4-6: Hessian eigenvalues at multiple scales
for i = 1:length(scales)
    sigma = scales(i);
    smoothed = imgaussfilt(img, sigma);
    [Ixx, Ixy, Iyy] = computeHessian(smoothed);
    
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    features(:, :, 3+i) = lambda1;
end

% Feature 7-9: Local binary pattern variants
features(:, :, 7) = extractLBP(img, 8, 1);
features(:, :, 8) = extractLBP(img, 8, 2);
features(:, :, 9) = extractLBP(img, 16, 2);

% Feature 10-12: Intensity statistics
features(:, :, 10) = img;
features(:, :, 11) = imfilter(img, fspecial('average', 5));
features(:, :, 12) = stdfilt(img, ones(5));

% Feature 13-15: Gabor filter responses
for i = 1:3
    theta = (i-1) * 60;
    gaborFilter = gabor(4, theta);
    gaborResponse = imfilter(img, real(gaborFilter), 'symmetric');
    features(:, :, 12+i) = gaborResponse;
end

% Feature 16-18: Vessel-specific measures
features(:, :, 16) = computeVesselnessFilter(img, [1 2 4]);
features(:, :, 17) = imfilter(img, fspecial('laplacian'));
features(:, :, 18) = imgradient(img);

% Feature 19-20: Texture measures
features(:, :, 19) = rangefilt(img, ones(5));
features(:, :, 20) = entropyfilt(img, ones(5));

% Reshape to pixel-wise feature vectors
features = reshape(features, rows*cols, numFeatures);
end

% Helper functions (same as in trainSVM)
function [Ixx, Ixy, Iyy] = computeHessian(img)
[Ix, Iy] = imgradientxy(img);
[Ixx, ~] = imgradientxy(Ix);
[Ixy, Iyy] = imgradientxy(Iy);
end

function lbp = extractLBP(img, neighbors, radius)
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
vesselness = zeros(size(img));

for scale = scales
    smoothed = imgaussfilt(img, scale);
    [Ixx, Ixy, Iyy] = computeHessian(smoothed);
    
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    lambda2 = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    
    vesselness = max(vesselness, -lambda2 .* (lambda2 < 0));
end
end
