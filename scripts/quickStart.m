% =================================================================
% Retinal Blood Vessel Segmentation - Quick Start Script
% =================================================================
%
% This script demonstrates how to use the retinal vessel segmentation
% framework with different approaches (supervised/unsupervised/hybrid).
%
% Author: Md Abu Sayed
% Date: February 2026
% =================================================================

clear; clc; close all;

%% 1. Setup Environment
fprintf('🔧 Setting up environment...\n');
addPaths(); % Add all necessary paths

%% 2. Configuration
config = struct();
config.windowSize = 15;        % Window size for line detection
config.patchSize = 32;         % Patch size for feature extraction
config.numTrees = 50;          % Number of trees in Random Forest
config.noiseThreshold = 100;   % Noise filtering threshold

fprintf('⚙️  Configuration loaded\n');

%% 3. Dataset Selection
% Available datasets: 'DRIVE', 'STARE', 'CHASEDB1'
dataset = 'DRIVE';
fprintf('📊 Selected dataset: %s\n', dataset);

%% 4. Choose Method
fprintf('\n🤖 Available Methods:\n');
fprintf('1. Unsupervised (Multi-scale line detection only)\n');
fprintf('2. Supervised (Random Forest classification)\n');
fprintf('3. Hybrid (Combination of both)\n');
method = input('Choose method (1-3): ');

switch method
    case 1
        fprintf('🔍 Using Unsupervised Method\n');
        methodName = 'unsupervised';
    case 2
        fprintf('🧠 Using Supervised Method\n');
        methodName = 'supervised';
    case 3
        fprintf('🔄 Using Hybrid Method\n');
        methodName = 'hybrid';
    otherwise
        error('❌ Invalid method selection');
end

%% 5. Sample Usage
fprintf('\n📝 Example usage for selected method:\n');

if method == 1
    % Unsupervised method example
    fprintf('💡 Unsupervised segmentation example:\n');
    fprintf('   img = imread(''path/to/fundus/image.jpg'');\n');
    fprintf('   mask = imread(''path/to/fov/mask.png'');\n');
    fprintf('   segmented = VesselSegment(img, mask);\n');
    
elseif method == 2
    % Supervised method example  
    fprintf('💡 Supervised segmentation example:\n');
    fprintf('   %% First train the model:\n');
    fprintf('   trainRFC; %% Run training script\n');
    fprintf('   \n');
    fprintf('   %% Then test:\n');
    fprintf('   testRFC; %% Run testing script\n');
    
else
    % Hybrid method example
    fprintf('💡 Hybrid segmentation example:\n');
    fprintf('   %% 1. Get initial segmentation (unsupervised)\n');
    fprintf('   img = imread(''path/to/fundus/image.jpg'');\n');
    fprintf('   mask = imread(''path/to/fov/mask.png'');\n');
    fprintf('   initial_seg = multi_test(img, mask);\n');
    fprintf('   \n');
    fprintf('   %% 2. Extract features and classify (supervised)\n');
    fprintf('   [features] = create_descriptor(img, initial_seg, 32);\n');
    fprintf('   %% Apply trained classifier for refinement\n');
end

%% 6. Performance Evaluation
fprintf('\n📈 To evaluate performance:\n');
fprintf('   accuracy_tesst; %% Run evaluation script\n');

%% 7. Tips
fprintf('\n💡 Tips:\n');
fprintf('   • Ensure datasets are in Images/RFC SET/ directory\n');
fprintf('   • Adjust parameters in configuration section above\n');
fprintf('   • Check Publications/ folder for research papers\n');
fprintf('   • Use addPaths() at start of any new MATLAB session\n');

fprintf('\n✅ Quick start guide complete!\n');
fprintf('📚 For detailed documentation, see README.md\n');
