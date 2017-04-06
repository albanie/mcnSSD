function setup_mcnSSD
%SETUP_MCNSSD Sets up mcnSSD by adding its folders to the MATLAB path

root = fileparts(mfilename('fullpath')) ;
addpath(genpath(root)) ;
