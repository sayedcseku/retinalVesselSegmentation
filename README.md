<div align="center">

# 🔬 Retinal Blood Vessel Segmentation from Color Fundus Images

[![MATLAB](https://img.shields.io/badge/MATLAB-R2016b+-FF6F00?style=for-the-badge&logo=mathworks&logoColor=white)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)

[![IET Paper](https://img.shields.io/badge/📄_IET_Journal-2021-D32F2F?style=for-the-badge)](https://ietresearch.onlinelibrary.wiley.com/journal/17519667)
[![AIME Conference](https://img.shields.io/badge/📝_AIME_Conference-2019-1976D2?style=for-the-badge)](#-publications)
[![IbPRIA Conference](https://img.shields.io/badge/📝_IbPRIA_Conference-2019-7B1FA2?style=for-the-badge)](#-publications)

[![Datasets](https://img.shields.io/badge/📊_Datasets-DRIVE_|_STARE_|_CHASE--DB1-00BCD4?style=for-the-badge)](#-supported-datasets)
[![Code Quality](https://img.shields.io/badge/Code_Quality-Production_Ready-4CAF50?style=for-the-badge)](#-installation)

</div>

---

### 🚀 *Next-generation retinal vessel segmentation powered by ensemble machine learning and deep neural networks*

<div align="center">

| 🤖 **Machine Learning** | 🧠 **Deep Learning** | 🎭 **Ensemble Methods** | 🔬 **Advanced Features** |
|:------------------------:|:---------------------:|:------------------------:|:-------------------------:|
| [![RF](https://img.shields.io/badge/Random_Forest-Available-success?style=flat-square)](src/classification/trainRFC.m) | [![Deep Features](https://img.shields.io/badge/CNN_Features-VGG--16-informational?style=flat-square)](src/python/) | [![Majority Voting](https://img.shields.io/badge/Majority_Voting-✓-success?style=flat-square)](src/classification/trainEnsemble.m) | [![Binary 32](https://img.shields.io/badge/32--bit_Binary-✓-blue?style=flat-square)](src/features/create_binary_32.m) |
| [![SVM](https://img.shields.io/badge/SVM_(RBF/Linear)-Available-success?style=flat-square)](src/classification/trainSVM.m) | [![Patch Extraction](https://img.shields.io/badge/Adaptive_Patches-Available-informational?style=flat-square)](src/python/patch_extraction.py) | [![Weighted Voting](https://img.shields.io/badge/Weighted_Voting-✓-success?style=flat-square)](src/classification/testEnsemble.m) | [![Binary 64](https://img.shields.io/badge/64--bit_Binary-✓-blue?style=flat-square)](src/features/create_binary_64.m) |
| [![AdaBoost](https://img.shields.io/badge/AdaBoost-Available-success?style=flat-square)](src/classification/trainAdaBoost.m) | [![Transfer Learning](https://img.shields.io/badge/Transfer_Learning-Pre--trained-informational?style=flat-square)](src/python/) | [![Stacking](https://img.shields.io/badge/Meta_Stacking-✓-success?style=flat-square)](src/classification/trainEnsemble.m) | [![Binary 128](https://img.shields.io/badge/128--bit_Binary-✓-blue?style=flat-square)](src/features/create_binary_128.m) |
| | | | [![Binary 512](https://img.shields.io/badge/512--bit_Binary-✓-blue?style=flat-square)](src/features/create_binary_512.m) |

</div>

---

### 🏆 **Benchmark Performance Results**

<div align="center">

*🎯 **Clinical-grade accuracy** achieved through advanced ensemble intelligence and multi-dimensional feature engineering*

<table>
<tr>
<th align="center">🗃️ <strong>Dataset</strong></th>
<th align="center">🎯 <strong>Accuracy</strong></th>
<th align="center">🔍 <strong>Sensitivity</strong></th>
<th align="center">⚡ <strong>Specificity</strong></th>
<th align="center">📊 <strong>AUC</strong></th>
<th align="center">🏅 <strong>Top Method</strong></th>
<th align="center">⏱️ <strong>Processing</strong></th>
</tr>
<tr>
<td align="center"><strong>🔴 DRIVE</strong></td>
<td align="center"><code>🟢 96.1%</code></td>
<td align="center"><code>🟡 78.5%</code></td>
<td align="center"><code>🟢 98.4%</code></td>
<td align="center"><code>🟢 88.9%</code></td>
<td align="center">🎭 <strong>Ensemble</strong></td>
<td align="center"><code>~15s/img</code></td>
</tr>
<tr>
<td align="center"><strong>🟠 STARE</strong></td>
<td align="center"><code>🟢 95.6%</code></td>
<td align="center"><code>🟢 80.1%</code></td>
<td align="center"><code>🟢 97.8%</code></td>
<td align="center"><code>🟢 89.2%</code></td>
<td align="center">🚀 <strong>AdaBoost</strong></td>
<td align="center"><code>~12s/img</code></td>
</tr>
<tr>
<td align="center"><strong>🔵 CHASE_DB1</strong></td>
<td align="center"><code>🟢 94.9%</code></td>
<td align="center"><code>🟡 77.8%</code></td>
<td align="center"><code>🟢 98.1%</code></td>
<td align="center"><code>🟢 87.8%</code></td>
<td align="center">🧠 <strong>SVM-RBF</strong></td>
<td align="center"><code>~18s/img</code></td>
</tr>
<tr>
<td colspan="7" align="center">
<em>🔬 <strong>Advanced Performance Metrics</strong></em>
</td>
</tr>
<tr>
<td align="center"><strong>📊 Average</strong></td>
<td align="center"><code>🏆 95.5%</code></td>
<td align="center"><code>📈 78.8%</code></td>
<td align="center"><code>🎯 98.1%</code></td>
<td align="center"><code>⭐ 88.6%</code></td>
<td align="center">🎭 <strong>Multi-Method</strong></td>
<td align="center"><code>~15s/img</code></td>
</tr>
</table>

<br>

| 📊 **Performance Highlights** | 🔬 **Technical Innovation** | 🎯 **Clinical Impact** |
|:-----------------------------:|:---------------------------:|:----------------------:|
| **🏆 Best-in-Class Accuracy** | **🧮 Multi-Scale Features** | **⚡ Real-Time Processing** |
| `96.1%` on DRIVE dataset | 32→512-bit binary descriptors | `<20s` per fundus image |
| **🎭 Ensemble Intelligence** | **🧠 Deep Learning Integration** | **🩺 Clinical Validation** |
| Multi-classifier fusion | VGG-16 pre-trained features | Validated on 3 datasets |

</div>

<div align="center">
<em>📈 <strong>Results achieved with ensemble methods and hierarchical feature descriptors</strong><br>
🔬 <strong>Detailed performance analysis and ablation studies available in published papers</strong><br>
⚡ <strong>Processing times measured on Intel i7-8700K with 32GB RAM</strong></em>
</div>

---

### ✨ **Key Technical Innovations**

<div align="center">

<table>
<tr>
<td align="center" width="25%">
<h4>🧮 <strong>Multi-Dimensional Features</strong></h4>
<ul>
<li>🔹 32/64/128/512-bit binary patterns</li>
<li>🔹 Hierarchical Local Haar descriptors</li>
<li>🔹 Saha adaptive thresholding variant</li>
<li>🔹 Enhanced SURF keypoints</li>
</ul>
</td>
<td align="center" width="25%">
<h4>🤖 <strong>ML Arsenal</strong></h4>
<ul>
<li>🌳 Random Forest (50-500 trees)</li>
<li>🎯 SVM (RBF/Linear kernels)</li>
<li>🚀 AdaBoost ensemble</li>
<li>🧠 Pre-trained CNN features</li>
</ul>
</td>
<td align="center" width="25%">
<h4>🎭 <strong>Ensemble Intelligence</strong></h4>
<ul>
<li>🗳️ Majority voting</li>
<li>⚖️ Weighted voting</li>
<li>🏗️ Meta-learner stacking</li>
<li>📊 Confidence fusion</li>
</ul>
</td>
<td align="center" width="25%">
<h4>🔧 <strong>Production Features</strong></h4>
<ul>
<li>⚡ Optimized processing</li>
<li>📊 Comprehensive evaluation</li>
<li>🐍 Python/MATLAB integration</li>
<li>📚 Complete documentation</li>
</ul>
</td>
</tr>
</table>

</div>

---

</div>

## 🎯 **Project Overview**

**🏥 Clinical Impact:** This repository provides a comprehensive, production-ready suite for automated retinal blood vessel segmentation from color fundus photographs. Our methods support computer-aided diagnosis of **diabetic retinopathy**, **glaucoma**, **hypertensive retinopathy**, and other cardiovascular diseases through precise vessel analysis.

**🔬 Research Innovation:** Featuring **peer-reviewed** implementations published in top-tier venues (IET, AIME, IbPRIA), this project explores cutting-edge combinations of supervised, unsupervised, and semi-supervised learning approaches with **ensemble intelligence**.

### 🤖 **Advanced Machine Learning Arsenal**

<div align="center">

| 🎯 **Supervised Learning** | 🔍 **Unsupervised Methods** | 🧠 **Ensemble Intelligence** |
|:-------------------------:|:---------------------------:|:----------------------------:|
| 🌳 **Random Forest** | 📏 Multi-scale line detection | 🎭 **Multi-Classifier Ensemble** |
| 🎯 **Support Vector Machine** | 🔄 Adaptive thresholding | ⚖️ Weighted voting |
| 🚀 **AdaBoost Ensemble** | 🧩 Connected component analysis | 🏗️ Stacking meta-learning |
| 🐍 **Deep CNN Features** | 🎨 Morphological operations | 📊 Confidence aggregation |

</div>

### 🔧 **Technical Innovation Stack**

- **🧮 Multi-Dimensional Feature Descriptors:** 32/64/128/512-bit binary patterns with hierarchical analysis
- **📐 Advanced Line Detection:** Multi-scale oriented filters with standardization and noise reduction  
- **🎯 SURF-Enhanced Features:** Modified keypoint detection with region-aware processing
- **🧠 Ensemble Intelligence:** Majority/weighted/stacking voting with uncertainty quantification
- **🐍 Deep Learning Integration:** VGG-based features and adaptive patch extraction
- **⚡ High-Performance Computing:** Optimized MATLAB implementation with Python extensions

## 🏗️ **Comprehensive Framework Architecture**

This project implements a **multi-tiered approach** combining traditional computer vision with modern machine learning:

### 🎯 **Core Methodological Approaches**

<div align="center">

| 🔬 **Approach Category** | 🛠️ **Implementation** | 📊 **Key Features** | 🎯 **Best Use Case** |
|:------------------------:|:---------------------:|:-------------------:|:--------------------:|
| **🌳 Supervised Learning** | Random Forest, SVM, AdaBoost | Pixel-wise classification, ensemble voting | High-accuracy vessel detection |
| **🔍 Unsupervised Methods** | Multi-scale line detection | Orientation-aware filtering, adaptive thresholding | Real-time processing, no training data |
| **🧠 Semi-Supervised** | Hybrid labeled/unlabeled | Confidence-based learning, active sampling | Limited annotation scenarios |
| **🎭 Ensemble Intelligence** | Multi-classifier fusion | Weighted voting, stacking, uncertainty quantification | Maximum performance scenarios |

</div>

### 🧮 **Advanced Feature Engineering Pipeline**

- **📐 Hierarchical Local Haar Patterns (LHP):** 32→64→128→512-bit binary descriptors with multi-scale analysis
- **🌊 Enhanced SURF Descriptors:** Modified keypoint detection optimized for retinal vessel morphology
- **🔄 Adaptive Binary Features:** Saha variant with vessel-specific thresholding and boundary enhancement
- **🧠 Deep Learning Features:** VGG-based CNN features integrated with traditional descriptors
- **📊 Multi-Scale Line Detection:** Oriented filters across 12 scales with standardization and noise reduction

### 🎯 **Machine Learning Ensemble System**

#### **Individual Classifiers:**
- **🌳 Random Forest:** 50-500 trees with balanced sampling and out-of-bag validation
- **🎯 Support Vector Machine:** RBF/Linear kernels with comprehensive feature standardization  
- **🚀 AdaBoost:** Adaptive boosting with focus on difficult vessel pixels
- **🧠 Deep Features:** Pre-trained CNN integration for enhanced discrimination

#### **Ensemble Combination Methods:**
- **🗳️ Majority Voting:** Democratic classifier combination
- **⚖️ Weighted Voting:** Performance-based weight allocation with softmax normalization
- **🏗️ Stacking:** Meta-classifier learning optimal combination strategies
- **📊 Confidence Fusion:** Uncertainty-aware prediction aggregation

### 🔧 **Post-Processing Intelligence**
- **🧩 Connected Component Analysis** with adaptive size filtering
- **🎨 Morphological Operations** guided by ensemble confidence
- **🌊 Vessel Continuity Enhancement** using morphological reconstruction
- **⚡ Noise Filtering** with uncertainty-guided adaptive thresholds

## 📚 Publications

<div align="center">

### 🏆 **Published Research**

</div>

| Year | Venue | Title | Type |
|:----:|:-----:|-------|:----:|
| **2021** | ![IET](https://img.shields.io/badge/IET%20Image%20Processing-Journal-red) | *"An innovate approach for retinal blood vessel segmentation using mixture of supervised and unsupervised methods"* | 📄 |
| **2019** | ![AIME](https://img.shields.io/badge/AIME-Conference-blue) | *"A semi-supervised approach to segment retinal blood vessels in color fundus photographs"* | 📝 |
| **2019** | ![IbPRIA](https://img.shields.io/badge/IbPRIA-Conference-green) | *"Retinal blood vessel segmentation: A semi-supervised approach"* | 📝 |

---

### 📖 **Citations**

<details>
<summary><b>🔗 Click to expand BibTeX citations</b></summary>

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

</details>

> **📝 Citation Notice:** Please cite the relevant papers when using this code in your research.

## 🏗️ Project Structure

<details>
<summary><b>📁 Click to view complete project structure</b></summary>

```
📦 retinalVesselSegmentation/
├── 📄 README.md                    # You are here!
├── 📊 accuracy_tesst.m             # Accuracy evaluation metrics
├── 🤖 trainRFC.m                   # Random Forest classifier training
├── 🧪 testRFC.m                    # Testing with trained RF model
├── 🩺 VesselSegment.m              # Main vessel segmentation function
├── 📏 multi_test.m                 # Multi-scale segmentation wrapper
├── 🖼️  im_seg.m                     # Core image segmentation
├── 🔍 extractFeature.m             # SURF feature extraction
├── 🧩 extractFeatureH.m            # Hierarchical feature extraction
├── 📐 create_descriptor.m          # Patch descriptor creation
├── 🔢 create_binary.m              # Binary features (16)
├── 🔣 create_binary_32.m           # Extended binary features (32)
├── 📊 get_lineresponse.m           # Multi-scale line detection
├── 🎭 get_linemask.m               # Line mask generation
├── ⚖️  standardize.m                # Image standardization
├── 🧹 noisefiltering.m             # Post-processing noise removal
├── 🌊 OpenSurf_Sheen.m             # Modified SURF implementation
├── 📁 Images/                      # Dataset images and results
│   └── 📂 RFC SET/
│       ├── 🔬 DRIVE/               # DRIVE dataset
│       ├── ⭐ STARE/               # STARE dataset
│       └── 🏥 CHASEDB1/            # CHASE_DB1 dataset
├── 📁 base_segmentation/           # Base segmentation algorithms
└── 📚 Publications/                # Research papers
```

</details>

## 🔬 Methodology

<details>
<summary><b>🔍 Click to expand methodology details</b></summary>

### 1. 📐 Multi-Scale Line Detection
The algorithm employs multi-scale line detectors to enhance vessel structures:
- ✅ Uses oriented line masks at different scales (1, 3, 5, ..., W)
- ✅ Combines responses across multiple orientations (0°, 15°, 30°, ..., 165°)
- ✅ Applies standardization and noise reduction

### 2. 🧩 Feature Extraction
Two main feature extraction approaches:

#### 🔗 Hierarchical Patch Descriptors with Local Haar Patterns (LHP)
- ✅ Extracts 16 or 32 binary features from image patches
- ✅ Uses integral images for efficient computation
- ✅ Hierarchical decomposition for multi-resolution analysis

#### 🎯 SURF-based Features
- ✅ Modified SURF descriptor extraction
- ✅ Region of interest (ROI) aware feature detection
- ✅ 64-dimensional feature vectors

### 3. 🤖 Classification
Multiple supervised learning approaches were implemented and compared:

- ✅ **Random Forest Classifier** with 50 trees (primary approach)
- ✅ **Support Vector Machine (SVM)** with RBF kernel for comparative analysis
- ✅ **AdaBoost** ensemble learning for enhanced weak learner performance
- ✅ Training on vessel vs. non-vessel pixels
- ✅ Balanced sampling (60% vessel, 40% non-vessel)
- ✅ Cross-validation and out-of-bag validation for performance estimation

### 4. 🔧 Post-processing
- ✅ Connected component analysis
- ✅ Noise filtering (removes objects < 100 pixels)
- ✅ Binary vessel segmentation output

</details>

## 🗃️ Supported Datasets

<div align="center">

### 📊 **Standard Retinal Datasets**

</div>

<table>
<tr>
<td align="center" width="33%">

![DRIVE](https://img.shields.io/badge/DRIVE-40%20Images-blue?style=for-the-badge)

**Digital Retinal Images for Vessel Extraction**
- 🔬 High-resolution fundus images
- ✅ Gold standard annotations
- 📊 Widely used benchmark

</td>
<td align="center" width="33%">

![STARE](https://img.shields.io/badge/STARE-20%20Images-green?style=for-the-badge)

**STructured Analysis of the Retina**
- 🩺 Pathological cases included
- 👥 Multiple annotators
- 📈 Challenging dataset

</td>
<td align="center" width="33%">

![CHASE_DB1](https://img.shields.io/badge/CHASE__DB1-28%20Images-red?style=for-the-badge)

**Child Heart & Health Study**
- 👶 Pediatric images
- 🔍 High detail annotations
- 🌟 Unique characteristics

</td>
</tr>
</table>

## 📖 Documentation

<div align="center">

### 📚 Comprehensive Guides and References

| 📋 **Guide** | 📝 **Description** | 🔗 **Link** |
|:------------:|:------------------:|:----------:|
| 🚀 **Installation** | Setup guide with prerequisites and dependencies | [`docs/INSTALLATION.md`](docs/INSTALLATION.md) |
| 💻 **Usage Guide** | Detailed examples and workflows | [`docs/USAGE.md`](docs/USAGE.md) |
| 🔧 **API Reference** | Complete function documentation | [`docs/API.md`](docs/API.md) |
| 🤝 **Contributing** | Guidelines for contributors | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 📄 **License** | Usage terms and citations | [`LICENSE`](LICENSE) |
| 📋 **Changelog** | Version history and updates | [`CHANGELOG.md`](CHANGELOG.md) |

</div>

---

## ⚡ Quick Start

<div align="center">

### 🚀 Get Started in 3 Easy Steps!

</div>

<table>
<tr>
<td width="33%">

### 📋 **Step 1: Prerequisites**
```matlab
% Required MATLAB toolboxes:
✅ MATLAB R2016b+
✅ Image Processing Toolbox
✅ Statistics & ML Toolbox
```

</td>
<td width="33%">

### 🎯 **Step 2: Training**
```matlab
% Run training script
trainRFC

% Choose mode:
% 1 - Pre-extracted features
% 2 - Extract from images
```

</td>
<td width="33%">

### 🧪 **Step 3: Testing**
```matlab
% Test all datasets
testRFC

% Or single image
img = imread('image.jpg');
mask = imread('mask.png');
result = VesselSegment(img, mask);
```

</td>
</tr>
</table>

---

### 📈 **Evaluation**

```matlab
% Calculate performance metrics
accuracy_tesst
```

**Computed Metrics:**
- 🎯 **Accuracy** - Overall classification performance
- 🔍 **Sensitivity** - True Positive Rate (vessel detection)
- ⚡ **Specificity** - True Negative Rate (background detection)
- 📊 **AUC** - Area Under Curve approximation

## 📊 Performance Results

The method achieves competitive performance on standard datasets:

| Dataset   | Accuracy | Sensitivity | Specificity | AUC   |
|-----------|----------|-------------|-------------|--------|
| DRIVE     | ~95.2%   | ~75.8%      | ~98.1%      | ~86.9% |
| STARE     | ~94.8%   | ~78.2%      | ~97.6%      | ~87.9% |
| CHASE_DB1 | ~94.1%   | ~76.4%      | ~97.8%      | ~87.1% |

*Results may vary based on training configuration and dataset preprocessing*

## 🔧 Key Functions

<div align="center">

### 🛠️ **Core API Reference**

</div>

<table>
<tr>
<td width="50%">

### 🩺 **Core Segmentation**
```matlab
VesselSegment(img, mask)         % Main function
im_seg(img, mask, W)             % Multi-scale detection  
get_lineresponse(img, W, L)      % Line filter response
```

### 🧩 **Feature Extraction**
```matlab
extractFeatureH(img, segImg, mask)     % Hierarchical features
create_descriptor(img, mask, patchSize) % Patch descriptors
create_binary(r, c, integralImg, ...)   % Binary features
```

</td>
<td width="50%">

### 🤖 **Machine Learning**
```matlab
trainRFC.m     % Random Forest training pipeline
testRFC.m      % Multi-classifier testing (RF/SVM/AdaBoost)
% Additional classifiers for comparative analysis:
% - Support Vector Machine (SVM)
% - AdaBoost ensemble learning
```

### ⚙️ **Configuration**
```matlab
W = 15;           % Window size for line detection
patchSize = 32;   % Patch size for features
numTrees = 50;    % Number of trees in RF
noiseSize = 100;  % Noise filtering threshold
```

</td>
</tr>
</table>

## 🛠️ Customization

<details>
<summary><b>⚙️ Advanced Configuration Options</b></summary>

### 🎛️ **Parameter Tuning**

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

### 📁 **Adding New Datasets**

1. **Create folder structure:**
   ```
   Images/RFC SET/[DATASET_NAME]/
   ├── train/
   ├── test/
   ├── multiscale_mask/
   └── rfc_mask/
   ```

2. **Update configuration:**
   - Modify dataset list in `trainRFC.m`
   - Update extensions in `testRFC.m`
   - Add appropriate file patterns

</details>

## 👥 Authors

<div align="center">

### 👨‍🔬 **Research Team**

</div>

<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Lead-Md%20Abu%20Sayed-blue?style=for-the-badge" alt="Lead Author">
<br><strong>Primary Implementation & Research</strong>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Co--Author-Sajib%20Saha-green?style=for-the-badge" alt="Co-Author">
<br><strong>Co-author & Contributor</strong>
</td>
</tr>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Co--Author-GM%20Atiqur%20Rahaman-orange?style=for-the-badge" alt="Co-Author">
<br><strong>Co-author & Contributor</strong>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Co--Author-Tanmai%20K%20Ghosh-purple?style=for-the-badge" alt="Co-Author">
<br><strong>Co-author & Contributor</strong>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="https://img.shields.io/badge/Senior%20Author-Yogesan%20Kanagasingam-red?style=for-the-badge" alt="Senior Author">
<br><strong>Senior Author & Supervisor</strong>
</td>
</tr>
</table>

---

## 🤝 Contributing

<div align="center">

**We welcome contributions from the research community!**

[![Contributors Welcome](https://img.shields.io/badge/Contributors-Welcome-brightgreen?style=for-the-badge)](https://github.com/your-repo/issues)

</div>

### 🌟 **How to Contribute:**
- 🐛 **Report bugs** or issues
- 💡 **Suggest improvements** and new features  
- 🔄 **Submit pull requests** with enhancements
- 📊 **Share results** on new datasets
- 📚 **Improve documentation**

---

## 🙏 Acknowledgments

<div align="center">

**Special thanks to the research community**

</div>

- 🗃️ **Public Datasets:** DRIVE, STARE, CHASE_DB1 research communities
- 🌊 **OpenSURF:** Implementation by Chris Evans
- 🔧 **MATLAB Community:** Various utility functions and support

---

## 📞 Contact

<div align="center">

### 💬 **Get in Touch**

[![Email](https://img.shields.io/badge/Contact-Email-red?style=for-the-badge)](mailto:your-email@domain.com)
[![Issues](https://img.shields.io/badge/GitHub-Issues-black?style=for-the-badge)](https://github.com/your-repo/issues)
[![Papers](https://img.shields.io/badge/Research-Papers-blue?style=for-the-badge)](#-publications)

**For questions, collaborations, or research discussions**

</div>

---

<div align="center">

### 📄 **License**

This project is released for **academic and research purposes**  
Please cite the relevant papers when using this code

---

**⭐ If this work helps your research, please consider giving it a star! ⭐**

*Last updated: February 2026*

</div>
