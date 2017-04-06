function run_ssd_tests(varargin)
% ------------------------
% run tests for SSD modules
% (based on vl_testnn)
% ------------------------

opts.cpu = true ;
opts.gpu = false ;
opts.single = true ;
opts.double = false ;
%opts.command = 'nn' ;
opts.command = 'ut' ; % test utils
opts = vl_argparse(opts, varargin) ;

import matlab.unittest.constraints.* ;
import matlab.unittest.selectors.* ;
import matlab.unittest.plugins.TAPPlugin;
import matlab.unittest.plugins.ToFile;

% pick tests
sel = HasName(StartsWithSubstring(opts.command)) ;
if ~opts.gpu
  sel = sel & ~HasName(ContainsSubstring('device=gpu')) ;
end
if ~opts.cpu
  sel = sel & ~HasName(ContainsSubstring('device=cpu')) ;
end
if ~opts.double
  sel = sel & ~HasName(ContainsSubstring('dataType=double')) ;
end
if ~opts.single
  sel = sel & ~HasName(ContainsSubstring('dataType=single')) ;
end

% add test class to path
addpath(genpath(fullfile(vl_rootnn, 'matlab', 'xtest'))) ;

ssdRoot = fullfile(vl_rootnn, 'examples', 'ssd') ;
addpath(genpath(ssdRoot)) ;

% run ssd-specific dev tests
ssdTestFolder = fullfile(ssdRoot, 'matlab/xtest/suite', 'dev') ;
%ssdTestFolder = fullfile(ssdRoot, 'matlab/xtest/suite') ;
suite = matlab.unittest.TestSuite.fromFolder(ssdTestFolder, sel) ;
runner = matlab.unittest.TestRunner.withTextOutput('Verbosity',3);
result = runner.run(suite);
display(result)
