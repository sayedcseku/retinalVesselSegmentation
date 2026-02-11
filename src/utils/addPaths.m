function addPaths()
% ADDPATHS - Add all necessary paths for the retinal vessel segmentation project
% 
% This function adds all required subdirectories to MATLAB's path to ensure
% all functions are accessible from anywhere in the project.
%
% Usage: addPaths()
%
% Author: Md Abu Sayed
% Date: February 2026

% Get the current file's directory (src/utils/)
currentDir = fileparts(mfilename('fullpath'));

% Get the project root directory (two levels up from src/utils/)
projectRoot = fileparts(fileparts(currentDir));

% Add all source directories to path
addpath(genpath(fullfile(projectRoot, 'src')));

% Add Images directory (for datasets)
addpath(genpath(fullfile(projectRoot, 'Images')));

% Add scripts directory
addpath(fullfile(projectRoot, 'scripts'));

fprintf('✅ All paths added successfully!\n');
fprintf('📁 Project root: %s\n', projectRoot);
fprintf('🔧 Source directories added to MATLAB path\n');

end
