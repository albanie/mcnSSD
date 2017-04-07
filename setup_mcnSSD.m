function setup_mcnSSD
%SETUP_MCNSSD Sets up mcnSSD by adding its folders to the MATLAB path

root = fileparts(mfilename('fullpath')) ;
addpath(root) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'core')) ;
addpath(fullfile(root, 'matlab/mex')) ;
