function [predictions, scores, performance] = testAdaBoost(model, testPath, groundTruthPath)
% TESTADABOOST Test trained AdaBoost model on retinal vessel segmentation
%
% This function applies a trained AdaBoost model to test images for retinal
% vessel segmentation and evaluates performance against ground truth masks.
% It provides comprehensive post-processing and performance analysis.
%
% Syntax:
%   predictions = testAdaBoost(model, testPath)
%   [predictions, scores] = testAdaBoost(model, testPath)
%   [predictions, scores, performance] = testAdaBoost(model, testPath, groundTruthPath)
%
% Inputs:
%   model           - Trained AdaBoost model from trainAdaBoost function
%   testPath        - Path to test images directory (string)
%   groundTruthPath - Path to ground truth masks directory (optional, string)
%
% Outputs:
%   predictions - Cell array of binary segmentation masks
%   scores      - Cell array of confidence/probability maps  
%   performance - Performance metrics structure (if ground truth provided)
%     .accuracy      - Overall pixel accuracy
%     .sensitivity   - Sensitivity (true positive rate)
%     .specificity   - Specificity (true negative rate)
%     .precision     - Precision (positive predictive value)
%     .f1Score       - F1-score (harmonic mean of precision and sensitivity)
%     .auc           - Area under ROC curve
%     .perImage      - Per-image performance metrics
%     .confusionMatrix - Overall confusion matrix
%
% Description:
%   This function performs comprehensive testing of AdaBoost-based vessel
%   segmentation including feature extraction, ensemble prediction, 
%   post-processing, and detailed performance evaluation. It leverages
%   AdaBoost's confidence scores for enhanced segmentation quality.
%
% Post-processing Features:
%   - Confidence-based thresholding
%   - Morphological refinement
%   - Connected component filtering
%   - Vessel continuity enhancement
%   - Adaptive threshold selection
%
% Example:
%   % Test without performance evaluation
%   predictions = testAdaBoost(trainedModel, 'Images/DRIVE/test/');
%   
%   % Test with comprehensive evaluation
%   [pred, scores, perf] = testAdaBoost(trainedModel, ...
%       'Images/DRIVE/test/', 'Images/DRIVE/masks/');
%   
%   % Display results
%   figure; 
%   subplot(1,2,1); imshow(predictions{1}); title('Segmentation');
%   subplot(1,2,2); imshow(scores{1}, []); title('Confidence Map');
%   fprintf('Overall F1-Score: %.3f\n', perf.f1Score);
%
% Performance Notes:
%   - AdaBoost confidence scores provide excellent segmentation quality
%   - Ensemble prediction is more robust than single classifiers
%   - Post-processing significantly improves vessel connectivity
%
% See also: trainAdaBoost, testRFC, testSVM, predict
%
% Reference: 
%   Sayed et al., "Retinal blood vessel segmentation using supervised and 
%   unsupervised approaches", IET Computer Vision, 2021
%
% Author: Retinal Vessel Segmentation Research Team
% Date: February 2026

%% Input validation
if nargin < 2
    error('testAdaBoost:NotEnoughInputs', 'At least 2 inputs required');
end

evaluatePerformance = (nargin >= 3) && ~isempty(groundTruthPath);

fprintf('🧪 Testing AdaBoost Model on Retinal Images\n');
fprintf('==========================================\n\n');

%% Load test images
fprintf('📂 Loading test images...\n');

imageFiles = dir(fullfile(testPath, '*.tif'));
if isempty(imageFiles)
    imageFiles = [dir(fullfile(testPath, '*.jpg')); ...
                  dir(fullfile(testPath, '*.png'))];
end

if isempty(imageFiles)
    error('testAdaBoost:NoImages', 'No test images found in %s', testPath);
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
        warning('testAdaBoost:MismatchedFiles', ...
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
    
    %% Extract features (same as training)
    features = extractAdaBoostFeaturesTest(img);
    
    % Apply feature standardization
    if ~isempty(model.featureStd)
        features = (features - model.featureStd.mean) ./ (model.featureStd.std + 1e-6);
    end
    
    % Remove invalid features
    validIdx = all(isfinite(features), 2);
    
    %% AdaBoost ensemble prediction
    pixelPredictions = zeros(size(features, 1), 1);
    pixelScores = zeros(size(features, 1), 2);
    
    if any(validIdx)
        try
            [pixelPredictions(validIdx), pixelScores(validIdx, :)] = ...
                predict(model.AdaModel, features(validIdx, :));
        catch ME
            warning('testAdaBoost:PredictionError', ...
                'Error predicting image %d: %s', i, ME.message);
            continue;
        end
    end
    
    %% Reshape predictions back to image format
    predictionMap = reshape(pixelPredictions, originalSize);
    confidenceMap = reshape(pixelScores(:, 2), originalSize); % Confidence of vessel class
    
    %% Advanced post-processing for AdaBoost
    processedPrediction = postProcessAdaBoostSegmentation(predictionMap, confidenceMap, img);
    
    %% Store results
    predictions{i} = processedPrediction;
    scores{i} = confidenceMap;
    
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
            metrics = calculateAdaBoostMetrics(processedPrediction, groundTruth, confidenceMap);
            
            performance.perImage.accuracy(i) = metrics.accuracy;
            performance.perImage.sensitivity(i) = metrics.sensitivity;
            performance.perImage.specificity(i) = metrics.specificity;
            performance.perImage.precision(i) = metrics.precision;
            performance.perImage.f1Score(i) = metrics.f1Score;
            performance.perImage.auc(i) = metrics.auc;
            
            % Accumulate for overall metrics
            allPredictions = [allPredictions; processedPrediction(:)];
            allGroundTruth = [allGroundTruth; groundTruth(:)];
            allScores = [allScores; confidenceMap(:)];
        else
            warning('testAdaBoost:NoGroundTruth', ...
                'No ground truth found for image %s', imgName);
        end
    end
end

%% Calculate overall performance metrics
if evaluatePerformance && ~isempty(allPredictions)
    fprintf('\n📈 Calculating overall performance metrics...\n');
    
    overallMetrics = calculateAdaBoostMetrics(allPredictions, allGroundTruth, allScores);
    
    performance.accuracy = overallMetrics.accuracy;
    performance.sensitivity = overallMetrics.sensitivity;
    performance.specificity = overallMetrics.specificity;
    performance.precision = overallMetrics.precision;
    performance.f1Score = overallMetrics.f1Score;
    performance.auc = overallMetrics.auc;
    
    % Confusion matrix
    performance.confusionMatrix = confusionmat(allGroundTruth, allPredictions);
    
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
    
    fprintf('\n📊 Confusion Matrix:\n');
    fprintf('   True Negative:  %d\n', performance.confusionMatrix(1,1));
    fprintf('   False Positive: %d\n', performance.confusionMatrix(1,2));
    fprintf('   False Negative: %d\n', performance.confusionMatrix(2,1));
    fprintf('   True Positive:  %d\n', performance.confusionMatrix(2,2));
end

fprintf('\n✅ AdaBoost testing completed successfully!\n');
fprintf('==========================================\n\n');

end

function processedPrediction = postProcessAdaBoostSegmentation(prediction, confidence, originalImg)
% Advanced post-processing for AdaBoost segmentation using confidence scores

% Convert to logical
prediction = logical(prediction);

%% Confidence-based refinement
% Use confidence scores to refine predictions
confThreshold = graythresh(confidence); % Otsu's threshold
highConfidenceVessels = confidence > (confThreshold * 1.2);
lowConfidenceBackground = confidence < (confThreshold * 0.8);

% Start with high-confidence predictions
refinedPrediction = highConfidenceVessels;

% Add medium-confidence regions that are connected to high-confidence vessels
mediumConfidence = (confidence >= confThreshold * 0.8) & (confidence <= confThreshold * 1.2);
connectedRegions = bwconncomp(mediumConfidence);

for i = 1:connectedRegions.NumObjects
    region = false(size(refinedPrediction));
    region(connectedRegions.PixelIdxList{i}) = true;
    
    % Check if this region is connected to high-confidence vessels
    dilated = imdilate(refinedPrediction, strel('disk', 2));
    if any(region(:) & dilated(:))
        refinedPrediction = refinedPrediction | region;
    end
end

%% Morphological refinement
% Opening to remove small noise
se1 = strel('disk', 1);
cleaned = imopen(refinedPrediction, se1);

% Closing to connect nearby vessel segments
se2 = strel('line', 3, 0); % Horizontal
se3 = strel('line', 3, 90); % Vertical
connected = imclose(imclose(cleaned, se2), se3);

%% Connected component analysis with size filtering
cc = bwconncomp(connected);
numPixels = cellfun(@numel, cc.PixelIdxList);

% Adaptive size threshold based on image size
imageSize = numel(originalImg);
minComponentSize = max(10, round(imageSize * 1e-5)); % 0.001% of image
maxComponentSize = round(imageSize * 0.1); % 10% of image (avoid large artifacts)

% Keep components within reasonable size range
validComponents = (numPixels >= minComponentSize) & (numPixels <= maxComponentSize);
processedPrediction = false(size(prediction));

for i = find(validComponents)
    processedPrediction(cc.PixelIdxList{i}) = true;
end

%% Vessel continuity enhancement
% Use morphological reconstruction to enhance vessel continuity
% Create markers from high-confidence vessels
markers = highConfidenceVessels & processedPrediction;
reconstructed = imreconstruct(markers, processedPrediction);

% Combine original and reconstructed
processedPrediction = processedPrediction | reconstructed;

%% Final cleanup
% Remove very small isolated components
cc_final = bwconncomp(processedPrediction);
numPixels_final = cellfun(@numel, cc_final.PixelIdxList);
finalMinSize = 5; % Very small threshold for final cleanup

for i = 1:cc_final.NumObjects
    if numPixels_final(i) < finalMinSize
        processedPrediction(cc_final.PixelIdxList{i}) = false;
    end
end

end

function metrics = calculateAdaBoostMetrics(prediction, groundTruth, scores)
% Calculate comprehensive performance metrics for AdaBoost

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
metrics.sensitivity = tp / max(tp + fn, 1); % Avoid division by zero
metrics.specificity = tn / max(tn + fp, 1);
metrics.precision = tp / max(tp + fp, 1);

if metrics.precision + metrics.sensitivity > 0
    metrics.f1Score = 2 * (metrics.precision * metrics.sensitivity) / ...
                     (metrics.precision + metrics.sensitivity);
else
    metrics.f1Score = 0;
end

% AUC calculation with error handling
if length(unique(scores)) > 1 && length(unique(groundTruth)) > 1
    try
        [~, ~, ~, metrics.auc] = perfcurve(groundTruth, scores, true);
    catch
        metrics.auc = 0.5; % Default AUC for problematic cases
    end
else
    metrics.auc = 0.5;
end

% Handle NaN values
fieldNames = {'sensitivity', 'specificity', 'precision', 'f1Score', 'auc'};
for i = 1:length(fieldNames)
    if isnan(metrics.(fieldNames{i}))
        if strcmp(fieldNames{i}, 'auc')
            metrics.(fieldNames{i}) = 0.5;
        else
            metrics.(fieldNames{i}) = 0;
        end
    end
end

end

function features = extractAdaBoostFeaturesTest(img)
% Extract the same features used in training (must match trainAdaBoost)
img = double(img);
[rows, cols] = size(img);

% Initialize feature matrix
numFeatures = 25; % Must match training exactly
features = zeros(rows, cols, numFeatures);

% Feature 1-4: Multi-scale Gaussian derivatives
scales = [0.5, 1, 2, 4];
for i = 1:length(scales)
    sigma = scales(i);
    [Gx, Gy] = imgradientxy(imgaussfilt(img, sigma));
    features(:, :, i) = sqrt(Gx.^2 + Gy.^2);
end

% Feature 5-8: Hessian-based vessel measures
for i = 1:length(scales)
    sigma = scales(i);
    smoothed = imgaussfilt(img, sigma);
    [Ixx, Ixy, Iyy] = computeHessianAdaBoostTest(smoothed);
    
    lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    lambda2 = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy).^2 + 4*Ixy.^2));
    vesselness = -lambda2 .* (lambda2 < 0 & abs(lambda1) < abs(lambda2));
    features(:, :, 4+i) = vesselness;
end

% Feature 9-12: Directional derivatives
directions = [0, 45, 90, 135];
for i = 1:length(directions)
    angle = directions(i) * pi / 180;
    kernel = [-sin(angle), 0, sin(angle); 
              -cos(angle), 0, cos(angle);
              -sin(angle), 0, sin(angle)] / 3;
    features(:, :, 8+i) = imfilter(img, kernel, 'symmetric');
end

% Feature 13-16: Local binary patterns
features(:, :, 13) = extractLBPAdaBoostTest(img, 8, 1);
features(:, :, 14) = extractLBPAdaBoostTest(img, 8, 2);
features(:, :, 15) = extractLBPAdaBoostTest(img, 16, 2);
features(:, :, 16) = extractLBPAdaBoostTest(img, 16, 3);

% Feature 17-19: Gabor filters
gabor_angles = [0, 45, 90];
for i = 1:length(gabor_angles)
    gaborFilter = gabor(2, gabor_angles(i));
    features(:, :, 16+i) = abs(imfilter(img, real(gaborFilter), 'symmetric'));
end

% Feature 20-22: Morphological operations
se1 = strel('disk', 1);
se2 = strel('disk', 2);
features(:, :, 20) = imopen(img, se1);
features(:, :, 21) = imclose(img, se1);
features(:, :, 22) = img - imopen(img, se2);

% Feature 23-25: Texture measures
features(:, :, 23) = rangefilt(img, ones(3));
features(:, :, 24) = stdfilt(img, ones(5));
features(:, :, 25) = entropyfilt(img, ones(5));

% Reshape to pixel-wise feature vectors
features = reshape(features, rows*cols, numFeatures);

% Remove any remaining NaN or Inf values
features(~isfinite(features)) = 0;
end

% Helper functions (same as in training)
function [Ixx, Ixy, Iyy] = computeHessianAdaBoostTest(img)
[Ix, Iy] = imgradientxy(img);
[Ixx, ~] = imgradientxy(Ix);
[Ixy, Iyy] = imgradientxy(Iy);
end

function lbp = extractLBPAdaBoostTest(img, neighbors, radius)
[rows, cols] = size(img);
lbp = zeros(rows, cols);

angles = 2 * pi * (0:neighbors-1) / neighbors;
neighbor_x = radius * cos(angles);
neighbor_y = radius * sin(angles);

for i = radius+1:rows-radius
    for j = radius+1:cols-radius
        center = img(i, j);
        pattern = 0;
        
        for k = 1:neighbors
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
                neighbor_val = center;
            end
            
            if neighbor_val >= center
                pattern = pattern + 2^(k-1);
            end
        end
        
        lbp(i, j) = pattern;
    end
end
end
