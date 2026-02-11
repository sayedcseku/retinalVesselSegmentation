function setupProject()
% SETUPPROJECT Initialize the retinal vessel segmentation project
%
% This function sets up the complete development environment for the
% retinal blood vessel segmentation project, including paths, dependencies,
% and initial configuration.
%
% Usage:
%   setupProject()
%
% Requirements:
%   - MATLAB R2016b or later
%   - Image Processing Toolbox
%   - Computer Vision Toolbox
%   - Statistics and Machine Learning Toolbox
%
% Author: Retinal Vessel Segmentation Team
% Date: February 2026

fprintf('\n🔬 Retinal Blood Vessel Segmentation Project Setup\n');
fprintf('=====================================================\n\n');

%% Check MATLAB Version
fprintf('📋 Checking MATLAB compatibility...\n');
matlabVersion = version('-release');
minRequiredYear = 2016;
currentYear = str2double(matlabVersion(1:4));

if currentYear < minRequiredYear
    warning('MATLAB R%db or later is recommended. Current: R%s', ...
        minRequiredYear, matlabVersion);
else
    fprintf('✅ MATLAB R%s is compatible\n', matlabVersion);
end

%% Check Required Toolboxes
fprintf('\n🔧 Checking required toolboxes...\n');
requiredToolboxes = {
    'Image Processing Toolbox', 'Image_Toolbox';
    'Computer Vision Toolbox', 'Computer_Vision_Toolbox';
    'Statistics and Machine Learning Toolbox', 'Statistics_Toolbox'
};

missingToolboxes = {};
for i = 1:size(requiredToolboxes, 1)
    toolboxName = requiredToolboxes{i, 1};
    toolboxCode = requiredToolboxes{i, 2};
    
    if license('test', toolboxCode)
        fprintf('✅ %s available\n', toolboxName);
    else
        fprintf('❌ %s not available\n', toolboxName);
        missingToolboxes{end+1} = toolboxName; %#ok<AGROW>
    end
end

if ~isempty(missingToolboxes)
    fprintf('\n⚠️  Warning: Some required toolboxes are missing:\n');
    for i = 1:length(missingToolboxes)
        fprintf('   - %s\n', missingToolboxes{i});
    end
    fprintf('Please install these toolboxes for full functionality.\n');
end

%% Setup Paths
fprintf('\n📁 Setting up project paths...\n');
try
    % Add all necessary paths
    addPaths();
    fprintf('✅ All project paths added successfully\n');
catch ME
    fprintf('❌ Error adding paths: %s\n', ME.message);
    return;
end

%% Check Dataset Structure
fprintf('\n📊 Checking dataset structure...\n');
datasetsPath = fullfile(pwd, 'Images', 'RFC SET');

if exist(datasetsPath, 'dir')
    datasets = {'DRIVE', 'STARE', 'CHASEDB1'};
    for i = 1:length(datasets)
        datasetPath = fullfile(datasetsPath, datasets{i});
        if exist(datasetPath, 'dir')
            fprintf('✅ %s dataset directory found\n', datasets{i});
        else
            fprintf('⚠️  %s dataset directory not found\n', datasets{i});
        end
    end
else
    fprintf('⚠️  Main dataset directory not found: %s\n', datasetsPath);
    fprintf('   Please create the directory and add datasets as described in docs/INSTALLATION.md\n');
end

%% Test Core Functionality
fprintf('\n🧪 Testing core functionality...\n');
try
    % Test if main functions are accessible
    if exist('VesselSegment', 'file') == 2
        fprintf('✅ Core segmentation function available\n');
    else
        fprintf('❌ Core segmentation function not found\n');
    end
    
    if exist('trainRFC', 'file') == 2
        fprintf('✅ Random Forest training function available\n');
    else
        fprintf('❌ Random Forest training function not found\n');
    end
    
    if exist('get_lineresponse', 'file') == 2
        fprintf('✅ Line detection function available\n');
    else
        fprintf('❌ Line detection function not found\n');
    end
    
catch ME
    fprintf('❌ Error during functionality test: %s\n', ME.message);
end

%% Memory Check
fprintf('\n💾 Checking system memory...\n');
try
    [~, memStats] = memory;
    availableGB = memStats.MemAvailableAllArrays / 1024^3;
    fprintf('✅ Available memory: %.1f GB\n', availableGB);
    
    if availableGB < 4
        fprintf('⚠️  Warning: Less than 4GB available memory.\n');
        fprintf('   Consider closing other applications for better performance.\n');
    end
catch
    fprintf('ℹ️  Could not determine memory status\n');
end

%% Final Setup
fprintf('\n🎯 Project setup summary:\n');
fprintf('----------------------------------------\n');
fprintf('📁 Project root: %s\n', pwd);
fprintf('🔧 MATLAB version: R%s\n', matlabVersion);
fprintf('📊 Toolboxes: %d/%d available\n', ...
    size(requiredToolboxes, 1) - length(missingToolboxes), ...
    size(requiredToolboxes, 1));

if isempty(missingToolboxes)
    fprintf('\n🎉 Setup completed successfully!\n');
    fprintf('👉 Next steps:\n');
    fprintf('   1. Review docs/INSTALLATION.md for dataset setup\n');
    fprintf('   2. Run scripts/quickStart.m for a demo\n');
    fprintf('   3. Check docs/USAGE.md for detailed examples\n');
else
    fprintf('\n⚠️  Setup completed with warnings.\n');
    fprintf('   Please install missing toolboxes for full functionality.\n');
end

fprintf('\n📚 Documentation available:\n');
fprintf('   • Installation: docs/INSTALLATION.md\n');
fprintf('   • Usage Guide: docs/USAGE.md\n');
fprintf('   • API Reference: docs/API.md\n');
fprintf('   • Contributing: CONTRIBUTING.md\n');

fprintf('\n🔬 Ready to start retinal vessel segmentation research!\n');
fprintf('=====================================================\n\n');

end
