# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-10

### Added
- 🎯 **Comprehensive repository restructuring** with organized source code directories
- 📚 **Enhanced documentation** with installation guide, usage examples, and API reference
- 🔧 **Utility functions** for path management and quick start guidance
- 🏗️ **Structured source code** organized by functionality (core, features, classification, preprocessing, evaluation)
- 📖 **Professional README** with badges, performance metrics, and citation information
- 🗂️ **Legacy code preservation** in dedicated legacy directory
- ⚡ **Quick start scripts** for new users and testing
- 📄 **License file** with proper attribution and citation requirements

### Changed
- 🔄 **File organization** - moved all MATLAB files to appropriate functional directories
- 📈 **Documentation format** - upgraded to professional markdown with enhanced formatting
- 🎨 **Visual enhancements** - added badges, tables, and improved layout
- 📊 **Performance reporting** - structured results in professional tables

### Project Structure
```
📁 retinalVesselSegmentation/
├── 📄 README.md                    # Comprehensive project documentation
├── 📄 LICENSE                      # MIT license with citation requirements
├── 📄 CHANGELOG.md                 # This changelog file
├── 📁 src/                         # Organized source code
│   ├── 📁 core/                    # Core segmentation algorithms
│   ├── 📁 features/                # Feature extraction methods
│   ├── 📁 classification/          # Machine learning classifiers
│   ├── 📁 preprocessing/           # Image preprocessing utilities
│   ├── 📁 evaluation/              # Performance evaluation tools
│   ├── 📁 utils/                   # Utility functions and path management
│   └── 📁 legacy/                  # Legacy code for backward compatibility
├── 📁 scripts/                     # Standalone scripts and examples
├── 📁 docs/                        # Documentation and guides
├── 📁 Images/                      # Dataset storage and results
└── 📁 Publications/                # Research papers and references
```

## [1.0.0] - 2021-XX-XX (Historical)

### Added
- 🔬 **Initial implementation** of retinal vessel segmentation algorithms
- 🤖 **Random Forest classification** for supervised vessel detection
- 🌊 **Multi-scale line detection** for unsupervised segmentation
- 🏗️ **Feature extraction** using SURF and custom descriptors
- 📊 **Evaluation metrics** for performance assessment
- 📚 **Research validation** on DRIVE, STARE, and CHASE_DB1 datasets

### Publications
- **IET Computer Vision (2021)**: Supervised and unsupervised approaches
- **AIME 2019**: Semi-supervised vessel segmentation approach
- **IbPRIA 2019**: Mixture of supervised and unsupervised methods

## Future Releases

### Planned Features [3.0.0]
- 🔮 **Deep learning integration** with CNN-based segmentation
- ⚡ **GPU acceleration** for faster processing
- 🌐 **Web interface** for easy access and testing
- 📱 **Mobile compatibility** for clinical applications
- 🔗 **API endpoints** for integration with medical systems

### Research Directions
- 🧠 **Advanced neural networks** (U-Net, DeepLab variants)
- 🔍 **Multi-modal fusion** with OCT and fluorescein angiography
- 📊 **Quantitative analysis** tools for clinical metrics
- 🔬 **3D vessel reconstruction** from multiple fundus views
