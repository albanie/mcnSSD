function setup_mcnSSD
%SETUP_MCNSSD Sets up mcnSSD by adding its folders to the MATLAB path

vl_setupnn ;

root = fileparts(mfilename('fullpath')) ;
addpath(root) ;
addpath(fullfile(root, 'core')) ;
addpath(fullfile(root, 'pascal')) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab/utils')) ;
addpath(fullfile(root, 'matlab/mex')) ;
