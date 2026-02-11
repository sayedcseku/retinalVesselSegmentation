# 🔬 API Reference

## Core Functions

### VesselSegment
Main function for retinal vessel segmentation.

```matlab
result = VesselSegment(imageFile, varargin)
```

**Parameters:**
- `imageFile` (string): Path to retinal fundus image
- `method` (optional): Segmentation method ('rf', 'multiscale', 'hybrid')
- `threshold` (optional): Threshold value for binary segmentation

**Returns:**
- `result` (struct): Segmentation results with fields:
  - `segmentation`: Binary vessel mask
  - `probability`: Probability map
  - `features`: Extracted features
  - `metrics`: Performance metrics (if ground truth available)

---

### multi_test
Batch processing function for multiple images.

```matlab
results = multi_test(dataset, method)
```

**Parameters:**
- `dataset` (string): Dataset name ('DRIVE', 'STARE', 'CHASEDB1')
- `method` (string): Segmentation method

**Returns:**
- `results` (cell): Array of segmentation results

---

## Feature Extraction

### extractFeature
Extract standard features from retinal images.

```matlab
features = extractFeature(image, mask)
```

**Parameters:**
- `image` (uint8): RGB retinal image
- `mask` (optional): Region of interest mask

**Returns:**
- `features` (double): Feature vector

---

### extractFeatureH
Extract hierarchical features using Local Haar Patterns.

```matlab
featuresH = extractFeatureH(image, patchSize)
```

**Parameters:**
- `image` (uint8): RGB retinal image
- `patchSize` (int): Size of local patches

**Returns:**
- `featuresH` (double): Hierarchical feature descriptors

---

## Classification Functions

### trainRFC
Train Random Forest classifier for vessel segmentation.

```matlab
model = trainRFC(trainingPath, groundTruthPath, options)
```

**Parameters:**
- `trainingPath` (string): Path to training images
- `groundTruthPath` (string): Path to ground truth masks
- `options` (struct): Training options
  - `numTrees`: Number of trees (default: 100)
  - `maxDepth`: Maximum tree depth (default: 10)

**Returns:**
- `model` (struct): Trained Random Forest model

---

### testRFC
Test Random Forest classifier on new images.

```matlab
predictions = testRFC(model, testPath)
```

**Parameters:**
- `model` (struct): Trained Random Forest model
- `testPath` (string): Path to test images

**Returns:**
- `predictions` (cell): Segmentation predictions

---

## Line Detection Functions

### get_lineresponse
Multi-scale line detection for vessel enhancement.

```matlab
response = get_lineresponse(image, scales, options)
```

**Parameters:**
- `image` (double): Grayscale retinal image
- `scales` (vector): Scale values for line detection
- `options` (struct): Detection options

**Returns:**
- `response` (double): Line response at multiple scales

---

### get_linemask
Generate binary vessel mask from line response.

```matlab
mask = get_linemask(response, threshold)
```

**Parameters:**
- `response` (double): Line response image
- `threshold` (double): Threshold value

**Returns:**
- `mask` (logical): Binary vessel mask

---

## Preprocessing Functions

### standardize
Standardize retinal image intensities.

```matlab
standardImage = standardize(image, method)
```

**Parameters:**
- `image` (uint8): Input retinal image
- `method` (string): Standardization method ('histogram', 'zscore')

**Returns:**
- `standardImage` (double): Standardized image

---

### noisefiltering
Apply noise filtering to retinal images.

```matlab
filteredImage = noisefiltering(image, filterType)
```

**Parameters:**
- `image` (uint8): Noisy retinal image
- `filterType` (string): Filter type ('gaussian', 'median', 'bilateral')

**Returns:**
- `filteredImage` (uint8): Filtered image

---

### fakepad
Add padding to images for boundary handling.

```matlab
paddedImage = fakepad(image, padSize, method)
```

**Parameters:**
- `image` (numeric): Input image
- `padSize` (int): Padding size
- `method` (string): Padding method ('symmetric', 'replicate')

**Returns:**
- `paddedImage` (numeric): Padded image

---

## Evaluation Functions

### accuracy_tesst
Calculate segmentation accuracy metrics.

```matlab
metrics = accuracy_tesst(prediction, groundTruth, mask)
```

**Parameters:**
- `prediction` (logical): Predicted vessel mask
- `groundTruth` (logical): Ground truth vessel mask
- `mask` (optional): Field of view mask

**Returns:**
- `metrics` (struct): Accuracy metrics
  - `accuracy`: Overall accuracy
  - `sensitivity`: Sensitivity (recall)
  - `specificity`: Specificity
  - `precision`: Precision
  - `f1Score`: F1-score
  - `auc`: Area under ROC curve

---

## Utility Functions

### addPaths
Add all necessary paths for the project.

```matlab
addPaths()
```

**Description:**
Automatically adds all source code directories to MATLAB path.

---

## Binary Descriptor Functions

### create_binary
Create binary descriptors for vessel detection.

```matlab
descriptors = create_binary(image, keypoints)
```

**Parameters:**
- `image` (uint8): Retinal image
- `keypoints` (struct): Detected keypoints

**Returns:**
- `descriptors` (logical): Binary descriptors

---

### create_binary_32
Create 32-bit binary descriptors.

```matlab
descriptors32 = create_binary_32(image, keypoints)
```

**Parameters:**
- `image` (uint8): Retinal image
- `keypoints` (struct): Detected keypoints

**Returns:**
- `descriptors32` (uint32): 32-bit binary descriptors

---

### create_descriptor
Create custom descriptors for vessel characterization.

```matlab
descriptors = create_descriptor(image, method, parameters)
```

**Parameters:**
- `image` (uint8): Retinal image
- `method` (string): Descriptor type ('lbp', 'glcm', 'gabor')
- `parameters` (struct): Method-specific parameters

**Returns:**
- `descriptors` (double): Feature descriptors

---

## Error Handling

All functions include comprehensive error checking:

```matlab
try
    result = VesselSegment('invalid_path.jpg');
catch ME
    switch ME.identifier
        case 'VesselSegment:FileNotFound'
            fprintf('Error: Image file not found\n');
        case 'VesselSegment:InvalidFormat'
            fprintf('Error: Unsupported image format\n');
        otherwise
            rethrow(ME);
    end
end
```

## Performance Considerations

### Memory Usage
- Large images may require processing in patches
- Use `clear` command to free memory between operations
- Monitor memory usage with `memory` command

### Processing Time
- Multi-scale operations are computationally intensive
- Consider parallel processing for batch operations
- Use appropriate image scaling for faster processing

### Optimization Tips
```matlab
% Efficient batch processing
parfor i = 1:length(imageList)
    results{i} = VesselSegment(imageList{i});
end

% Memory-efficient processing
for i = 1:length(imageList)
    result = VesselSegment(imageList{i});
    save(['result_' num2str(i) '.mat'], 'result');
    clear result; % Free memory
end
```
