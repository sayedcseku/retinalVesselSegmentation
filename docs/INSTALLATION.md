# 🚀 Installation Guide

## Prerequisites

### MATLAB Requirements
- **MATLAB R2016b or later** (recommended: R2019b+)
- **Required Toolboxes:**
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Statistics and Machine Learning Toolbox
  - Signal Processing Toolbox

### System Requirements
- **Memory:** Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage:** At least 5GB free space for datasets and results
- **OS:** Windows 10+, macOS 10.14+, or Ubuntu 18.04+

## Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/retinalVesselSegmentation.git
   cd retinalVesselSegmentation
   ```

2. **Setup MATLAB Paths**
   ```matlab
   % In MATLAB command window
   run('src/utils/addPaths.m')
   ```

3. **Quick Start**
   ```matlab
   % Run the quick start script
   run('scripts/quickStart.m')
   ```

## Dataset Setup

### Download Datasets
1. **DRIVE Database:** [Download from official source](https://drive.grand-challenge.org/)
2. **STARE Database:** [Download from official source](http://cecas.clemson.edu/~ahoover/stare/)
3. **CHASE_DB1:** [Download from official source](https://blogs.kingston.ac.uk/retinal/chasedb1/)

### Directory Structure
Place downloaded datasets in the `Images/RFC SET/` directory following this structure:
```
Images/RFC SET/
├── DRIVE/
│   ├── test/
│   └── train/
├── STARE/
│   ├── test/
│   └── train/
└── CHASEDB1/
    ├── test/
    └── train/
```

## Verification

Run the test script to verify installation:
```matlab
% Test the installation
run('scripts/Test.m')
```

## Troubleshooting

### Common Issues

1. **Path Issues**
   - Ensure all paths are added correctly using `addPaths.m`
   - Check MATLAB current directory

2. **Memory Issues**
   - Reduce image batch size in processing functions
   - Close unnecessary applications

3. **Toolbox Dependencies**
   - Verify all required toolboxes are installed
   - Check MATLAB license for toolbox availability

### Getting Help

- Check the [main README](../README.md) for detailed usage
- Review function documentation using `help functionName`
- Create an issue on GitHub for bugs or questions
