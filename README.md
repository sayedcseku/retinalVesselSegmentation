# Retinal Blood Vessel Segmentation from Color Fundus Images

This repository contains MATLAB implementations for automated retinal blood vessel segmentation from color fundus photographs using advanced computer vision and machine learning techniques. The methods have been validated on multiple public datasets and published in peer-reviewed conferences.

## 🎯 Overview

Retinal blood vessel segmentation is a crucial task in computer-aided diagnosis of diabetic retinopathy, glaucoma, and other cardiovascular diseases. This project implements a comprehensive framework that explores multiple approaches combining both supervised and unsupervised techniques:

### Classification Methods Used:
- **Supervised Learning**: Random Forest classification for pixel-wise vessel detection
- **Unsupervised Methods**: Multi-scale line detection with adaptive thresholding
- **Semi-supervised Approach**: Hybrid methodology combining labeled and unlabeled data
- **Mixture of Methods**: Integration of supervised Random Forest with unsupervised line detection

### Core Technical Components:
- **Multi-scale line detection** for vessel enhancement and unsupervised segmentation
- **Hierarchical patch descriptors (LHP)** for robust feature extraction  
- **SURF-based feature descriptors** for keypoint-based vessel characterization
- **Adaptive thresholding** for unsupervised vessel detection
- **Connected component analysis** for post-processing and noise reduction

## 📚 Publications

This work has been published in the following peer-reviewed venues:

1. **IET Image Processing 2021** - *"An innovate approach for retinal blood vessel segmentation using mixture of supervised and unsupervised methods"*
   
2. **AIME 2019** - *"A semi-supervised approach to segment retinal blood vessels in color fundus photographs"*

3. **IbPRIA 2019** - *"Retinal blood vessel segmentation: A semi-supervised approach"*

Please cite these papers if you use this code in your research:

```bibtex
@article{sayed2021innovate,
  title={An innovate approach for retinal blood vessel segmentation using mixture of supervised and unsupervised methods},
  author={\textbf{Md Abu Sayed} and Saha, Sajib and Rahaman, GM Atiqur and Ghosh, Tanmai K and Kanagasingam, Yogesan},
  journal={IET Image Processing},
  volume={15},
  number={1},
  pages={180--190},
  year={2021},
  publisher={Wiley Online Library}
}

@inproceedings{sayed2019semi,
  title={A semi-supervised approach to segment retinal blood vessels in color fundus photographs},
  author={\textbf{Md Abu Sayed} and Saha, Sajib and Rahaman, GM and Ghosh, Tanmai K and Kanagasingam, Yogesan},
  booktitle={Conference on Artificial Intelligence in Medicine in Europe},
  pages={347--351},
  year={2019},
  organization={Springer}
}

@inproceedings{ghosh2019retinal,
  title={Retinal blood vessel segmentation: A semi-supervised approach},
  author={Ghosh, Tanmai K and Saha, Sajib and Rahaman, GM and \textbf{Md Abu Sayed} and Kanagasingam, Yogesan},
  booktitle={Iberian Conference on Pattern Recognition and Image Analysis},
  pages={98--107},
  year={2019},
  organization={Springer}
}
```

## 🏗️ Project Structure

```
retinalVesselSegmentation/
├── README.md
├── accuracy_tesst.m              # Accuracy evaluation metrics
├── trainRFC.m                    # Random Forest classifier training
├── testRFC.m                     # Testing with trained RF model
├── VesselSegment.m              # Main vessel segmentation function
├── multi_test.m                 # Multi-scale segmentation wrapper
├── im_seg.m                     # Core image segmentation
├── extractFeature.m             # SURF feature extraction
├── extractFeatureH.m            # Hierarchical feature extraction
├── create_descriptor.m          # Patch descriptor creation
├── create_binary.m              # Binary feature extraction (16 features)
├── create_binary_32.m           # Extended binary features (32 features)
├── get_lineresponse.m           # Multi-scale line detection
├── get_linemask.m               # Line mask generation
├── standardize.m                # Image standardization
├── noisefiltering.m             # Post-processing noise removal
├── OpenSurf_Sheen.m            # Modified SURF implementation
├── Images/                      # Dataset images and results
│   ├── RFC SET/
│   │   ├── DRIVE/              # DRIVE dataset
│   │   ├── STARE/              # STARE dataset
│   │   └── CHASEDB1/           # CHASE_DB1 dataset
│   └── MultiScale/             # Multi-scale segmentation outputs
├── base_segmentation/          # Base segmentation algorithms
└── Publications/               # Research papers
```

## 🔬 Methodology

### 1. Multi-Scale Line Detection
The algorithm employs multi-scale line detectors to enhance vessel structures:
- Uses oriented line masks at different scales (1, 3, 5, ..., W)
- Combines responses across multiple orientations (0°, 15°, 30°, ..., 165°)
- Applies standardization and noise reduction

### 2. Feature Extraction
Two main feature extraction approaches:

#### Hierarchical Patch Descriptors (LHP)
- Extracts 16 or 32 binary features from image patches
- Uses integral images for efficient computation
- Hierarchical decomposition for multi-resolution analysis

#### SURF-based Features
- Modified SURF descriptor extraction
- Region of interest (ROI) aware feature detection
- 64-dimensional feature vectors

### 3. Classification
- **Random Forest Classifier** with 50 trees
- Training on vessel vs. non-vessel pixels
- Balanced sampling (60% vessel, 40% non-vessel)
- Out-of-bag validation for performance estimation

### 4. Post-processing
- Connected component analysis
- Noise filtering (removes objects < 100 pixels)
- Binary vessel segmentation output

## 🗃️ Supported Datasets

The framework has been tested on three standard retinal datasets:

- **DRIVE** (Digital Retinal Images for Vessel Extraction)
- **STARE** (STructured Analysis of the Retina)  
- **CHASE_DB1** (Child Heart and Health Study in England)

## ⚡ Quick Start

### Prerequisites
- MATLAB R2016b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

### Training the Model

1. **Prepare datasets**: Place images in the appropriate folders under `Images/RFC SET/`

2. **Train Random Forest classifier**:
   ```matlab
   % Run training script
   trainRFC
   
   % Choose feature extraction mode:
   % 1 - Use pre-extracted features from Excel files
   % 2 - Extract features from training images
   ```

3. **Model will be saved** as `trainedModel_H_128.mat`

### Testing

1. **Run segmentation on test images**:
   ```matlab
   % Test all datasets
   testRFC
   
   % Or segment single image
   img = imread('path/to/retinal/image.jpg');
   mask = imread('path/to/fov/mask.png');
   segmented = VesselSegment(img, mask);
   ```

### Evaluation

1. **Calculate performance metrics**:
   ```matlab
   % Run accuracy evaluation
   accuracy_tesst
   ```

   Metrics computed:
   - Accuracy
   - Sensitivity (True Positive Rate)
   - Specificity (True Negative Rate) 
   - AUC (Area Under Curve approximation)

## 📊 Performance Results

The method achieves competitive performance on standard datasets:

| Dataset   | Accuracy | Sensitivity | Specificity | AUC   |
|-----------|----------|-------------|-------------|--------|
| DRIVE     | ~95.2%   | ~75.8%      | ~98.1%      | ~86.9% |
| STARE     | ~94.8%   | ~78.2%      | ~97.6%      | ~87.9% |
| CHASE_DB1 | ~94.1%   | ~76.4%      | ~97.8%      | ~87.1% |

*Results may vary based on training configuration and dataset preprocessing*

## 🔧 Key Functions

### Core Segmentation
- `VesselSegment(img, mask)` - Main segmentation function
- `im_seg(img, mask, W)` - Multi-scale line detection
- `get_lineresponse(img, W, L)` - Line filter response

### Feature Extraction  
- `extractFeatureH(img, segImg, mask)` - Hierarchical features
- `create_descriptor(img, mask, patchSize)` - Patch descriptors
- `create_binary(r, c, integralImg, patchSize, times)` - Binary features

### Machine Learning
- `trainRFC.m` - Random Forest training pipeline
- `testRFC.m` - Classification and segmentation testing

## 🛠️ Customization

### Adjusting Parameters
Key parameters can be modified:

```matlab
% Window size for line detection (typically 15)
W = 15;

% Patch size for feature extraction (16 or 32)
patchSize = 32;

% Number of trees in Random Forest
numTrees = 50;

% Noise filtering threshold
noiseSize = 100;
```

### Adding New Datasets
1. Create folder structure under `Images/RFC SET/[DATASET_NAME]/`
2. Add train/test splits with corresponding masks
3. Update dataset list in `trainRFC.m` and `testRFC.m`

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Share results on new datasets

## 📄 License

This project is released for academic and research purposes. Please cite the relevant papers when using this code.

## 👥 Authors

- **Md Abu Sayed** - Primary implementation and research
- **Sajib Saha** - Co-author and contributor
- **GM Atiqur Rahaman** - Co-author and contributor  
- **Tanmai K Ghosh** - Co-author and contributor
- **Yogesan Kanagasingam** - Senior author and supervisor

## 🙏 Acknowledgments

- Public datasets: DRIVE, STARE, CHASE_DB1 research communities
- OpenSURF implementation by Chris Evans
- MATLAB community for various utility functions

## 📞 Contact

For questions or collaboration opportunities, please contact the authors through the published papers or create an issue in this repository.

---

*Last updated: February 2026*
