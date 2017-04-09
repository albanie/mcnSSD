function test_mcnSSD
% ------------------------
% run tests for SSD module
% ------------------------

% add tests to path
path = fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest') ;
addpath(path) ;

% test network layers
run_ssd_tests('command', 'nn') ;

% test utils
%run_ssd_tests('command', 'ut') ;
