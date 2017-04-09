function test_mcnSSD
% ------------------------
% run tests for SSD module
% ------------------------

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_ssd_tests('command', 'nn') ;

% test utils
run_ssd_tests('command', 'ut') ;
