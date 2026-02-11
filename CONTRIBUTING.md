# 🤝 Contributing to Retinal Vessel Segmentation

We welcome contributions to improve this retinal vessel segmentation project! Whether you're fixing bugs, adding features, improving documentation, or sharing research insights, your contributions are valuable.

## 🚀 Getting Started

### Prerequisites
- MATLAB R2016b or later
- Git knowledge
- Understanding of computer vision and medical image analysis

### Setting Up Development Environment

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/retinalVesselSegmentation.git
   cd retinalVesselSegmentation
   ```

2. **Create Development Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

3. **Setup MATLAB Environment**
   ```matlab
   % In MATLAB
   run('src/utils/addPaths.m');
   run('scripts/quickStart.m'); % Verify setup
   ```

## 📝 How to Contribute

### 🐛 Bug Reports
When reporting bugs, please include:
- MATLAB version and toolboxes
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Sample images (if applicable)

### 💡 Feature Requests
For new features, please:
- Check existing issues first
- Describe the use case
- Explain expected functionality
- Consider implementation approach

### 🔧 Code Contributions

#### Code Style Guidelines
- **MATLAB Conventions**: Follow standard MATLAB naming conventions
- **Function Documentation**: Include comprehensive help comments
- **Variable Naming**: Use descriptive names (e.g., `vesselMask` not `vm`)
- **Code Organization**: Keep functions focused and modular

#### Example Function Template
```matlab
function result = myFunction(inputImage, parameters)
% MYFUNCTION Brief description of what the function does
%
% Syntax:
%   result = myFunction(inputImage, parameters)
%
% Inputs:
%   inputImage - RGB retinal fundus image (uint8)
%   parameters - Structure with configuration options
%
% Outputs:
%   result - Structure containing segmentation results
%
% Example:
%   img = imread('retinal_image.tif');
%   params.threshold = 0.5;
%   result = myFunction(img, params);
%
% See also: relatedFunction1, relatedFunction2
%
% Author: Your Name
% Date: YYYY-MM-DD

% Input validation
if nargin < 2
    error('myFunction:NotEnoughInputs', 'At least 2 inputs required');
end

% Main function logic here
% ...

end
```

### 🧪 Testing Guidelines

#### Unit Tests
- Test new functions with sample data
- Include edge cases and error conditions
- Document test procedures

#### Integration Tests
- Test with all supported datasets
- Verify backward compatibility
- Check performance benchmarks

#### Example Test Structure
```matlab
function testMyFunction()
% TESTMYFUNCTION Unit tests for myFunction
%
% This function tests various scenarios for myFunction including:
% - Normal operation with valid inputs
% - Error handling with invalid inputs
% - Edge cases and boundary conditions

% Test 1: Normal operation
fprintf('Testing normal operation...\n');
testImage = imread('Images/RFC SET/DRIVE/test/01_test.tif');
params.threshold = 0.5;
result = myFunction(testImage, params);
assert(~isempty(result), 'Function should return non-empty result');

% Test 2: Error handling
fprintf('Testing error handling...\n');
try
    myFunction(); % Should error
    error('Expected error did not occur');
catch ME
    assert(strcmp(ME.identifier, 'myFunction:NotEnoughInputs'));
end

fprintf('All tests passed!\n');
end
```

## 📋 Contribution Process

### 1. Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/vessel-enhancement

# 2. Make changes and test
# ... develop and test your code ...

# 3. Commit with descriptive messages
git add .
git commit -m "feat: add vessel enhancement algorithm

- Implement multi-scale Hessian filtering
- Add adaptive threshold selection
- Include performance benchmarks
- Update documentation"

# 4. Push to your fork
git push origin feature/vessel-enhancement

# 5. Create Pull Request on GitHub
```

### 2. Commit Message Format
Use conventional commits format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(classification): add SVM classifier implementation
fix(preprocessing): resolve memory leak in image filtering
docs(readme): update installation instructions
test(core): add unit tests for VesselSegment function
```

### 3. Pull Request Guidelines
- **Clear Title**: Descriptive title explaining the change
- **Detailed Description**: What, why, and how of the changes
- **Testing**: Describe testing performed
- **Screenshots**: Include result images if applicable
- **Breaking Changes**: Highlight any breaking changes

#### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug Y
- [ ] Updated documentation Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests with DRIVE dataset
- [ ] Performance benchmarks maintained

## Screenshots/Results
(If applicable, include segmentation results)

## Breaking Changes
(Describe any breaking changes)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🔬 Research Contributions

### Academic Contributions
- **Novel Algorithms**: Implement new segmentation approaches
- **Performance Improvements**: Enhance existing methods
- **Dataset Extensions**: Support for new retinal image datasets
- **Comparative Studies**: Benchmarking against state-of-the-art methods

### Citation Requirements
If your contribution is based on published research, please:
- Add appropriate citations
- Update the Publications section
- Include BibTeX entries
- Respect intellectual property rights

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **CHANGELOG.md** with specific contributions
- **Academic publications** (for significant research contributions)
- **GitHub contributors** page

## ❓ Getting Help

### Community Support
- **GitHub Issues**: Technical questions and discussions
- **Email**: Contact maintainers for urgent issues
- **Documentation**: Check docs/ directory for guides

### Code Review Process
1. **Automated Checks**: Code style and basic functionality
2. **Peer Review**: Technical review by maintainers
3. **Testing**: Comprehensive testing on multiple datasets
4. **Documentation Review**: Ensure documentation is updated

## 📄 License
By contributing to this project, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Thank you for contributing to advancing retinal vessel segmentation research! 🙏**
