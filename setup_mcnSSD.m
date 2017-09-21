function setup_mcnSSD
%SETUP_MCNSSD Sets up mcnSSD by adding its folders to the MATLAB path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/misc']) ;
  addpath([root '/core'], [root '/pascal'], [root '/matlab']) ;
  addpath([root '/matlab/utils'], [root '/matlab/mex']) ;
