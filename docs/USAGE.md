# 📖 Usage Guide

## Basic Usage

### 1. Quick Start Example
```matlab
% Add paths and setup environment
run('src/utils/addPaths.m');

% Run quick start demo
run('scripts/quickStart.m');
```

### 2. Single Image Segmentation
```matlab
% Load and segment a single retinal image
imageFile = 'Images/RFC SET/DRIVE/test/01_test.tif';
result = VesselSegment(imageFile);

% Display results
figure;
imshow(result.segmentation);
title('Vessel Segmentation Result');
```

### 3. Batch Processing
```matlab
% Process multiple images from a dataset
dataset = 'DRIVE'; % Options: 'DRIVE', 'STARE', 'CHASEDB1'
results = multi_test(dataset);
```

## Advanced Usage

### Training Custom Models

#### Random Forest Classifier
```matlab
% Prepare training data
trainingPath = 'Images/RFC SET/DRIVE/train/';
groundTruthPath = 'Images/RFC SET/DRIVE/rfc_mask/';

% Train RFC model
model = trainRFC(trainingPath, groundTruthPath);

% Test the model
testPath = 'Images/RFC SET/DRIVE/test/';
results = testRFC(model, testPath);
```

#### Feature Extraction
```matlab
% Extract features from retinal images
imageFile = 'path/to/retinal/image.tif';

% Standard feature extraction
features = extractFeature(imageFile);

% Hierarchical feature extraction
featuresH = extractFeatureH(imageFile);

% Create binary descriptors
binaryDesc = create_binary(imageFile);
binaryDesc32 = create_binary_32(imageFile);
```

### Multi-scale Line Detection
```matlab
% Perform unsupervised vessel segmentation
imageFile = 'Images/RFC SET/DRIVE/test/01_test.tif';

% Get line response at multiple scales
lineResponse = get_lineresponse(imageFile);

% Generate line mask
lineMask = get_linemask(lineResponse);
```

### Preprocessing Options
```matlab
% Noise filtering
filteredImage = noisefiltering(originalImage);

% Image standardization
standardImage = standardize(originalImage);

% Fake padding for boundary handling
paddedImage = fakepad(originalImage, padSize);
```

## Performance Evaluation

### Accuracy Assessment
```matlab
% Evaluate segmentation accuracy
groundTruth = imread('path/to/ground/truth.png');
segmentation = imread('path/to/segmentation/result.png');

% Calculate accuracy metrics
metrics = accuracy_tesst(segmentation, groundTruth);

% Display results
fprintf('Accuracy: %.2f%%\n', metrics.accuracy);
fprintf('Sensitivity: %.2f%%\n', metrics.sensitivity);
fprintf('Specificity: %.2f%%\n', metrics.specificity);
```

## Configuration Options

### Dataset-Specific Parameters

#### DRIVE Dataset
- **Image Size:** 584 × 565 pixels
- **Training Images:** 20 images
- **Test Images:** 20 images
- **File Format:** TIFF

#### STARE Dataset
- **Image Size:** 700 × 605 pixels
- **Training Images:** 10 images
- **Test Images:** 10 images
- **File Format:** PPM

#### CHASE_DB1 Dataset
- **Image Size:** 999 × 960 pixels
- **Training Images:** 8 images
- **Test Images:** 20 images
- **File Format:** JPG

## Output Formats

The segmentation results are saved in multiple formats:

1. **Binary Masks:** `.png` format in `rfc_mask/` directories
2. **Processed Images:** Color-coded results in `rfc_output/` directories
3. **Multi-scale Results:** Intermediate results in `multiscale_mask/` directories

## Tips for Best Results

1. **Image Quality:** Ensure input images are high-quality and properly centered
2. **Preprocessing:** Apply appropriate noise filtering for better results
3. **Parameter Tuning:** Adjust threshold values based on dataset characteristics
4. **Memory Management:** Process large datasets in batches to avoid memory issues

## Common Workflows

### Research Workflow
```matlab
% 1. Setup environment
run('src/utils/addPaths.m');

% 2. Prepare datasets
% Place datasets in appropriate directories

% 3. Train models
model = trainRFC('DRIVE');

% 4. Test and evaluate
results = testRFC(model, 'DRIVE');
metrics = accuracy_tesst(results.segmentation, results.groundTruth);

% 5. Analyze results
% Generate plots and statistics
```

### Clinical Application Workflow
```matlab
% 1. Load clinical image
clinicalImage = 'path/to/clinical/fundus/image.tif';

% 2. Preprocess
processedImage = noisefiltering(imread(clinicalImage));

% 3. Segment vessels
segmentation = VesselSegment(processedImage);

% 4. Post-process and analyze
% Apply clinical analysis tools
```
