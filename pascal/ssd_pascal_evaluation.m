function ssd_pascal_evaluation(varargin)
%SSD_PASCAL_EVALUATION evaluate SSD detector on pascal VOC

opts.net = [] ;
opts.gpus = 1 ;
opts.evalVersion = 'fast' ;
opts.modelName = 'ssd-pascal-vggvd-300' ;
opts = vl_argparse(opts, varargin) ;

% load network
if isempty(opts.net)
  net = ssd_zoo(opts.modelName) ; 
else
  net = opts.net ; 
end

% evaluation options
opts.testset = 'test' ; 
opts.prefetch = true ;
opts.fixedSizeInputs = false ;

% configure batch opts
batchOpts.batchSize = 8 ;
batchOpts.numThreads = 4 ;
batchOpts.use_vl_imreadjpeg = true ; 
batchOpts.imageSize = net.meta.normalization.imageSize ;

% cache configuration 
cacheOpts.refreshPredictionCache = false ;
cacheOpts.refreshDecodedPredCache = false ;
cacheOpts.refreshEvaluationCache = false ;
cacheOpts.refreshFigures = false ;

% configure model options
modelOpts.predVar = 'detection_out' ;
modelOpts.get_eval_batch = @ssd_eval_get_batch ;

% configure dataset options
dataOpts.name = 'pascal' ;
dataOpts.decoder = 'serial' ;
dataOpts.getImdb = @getPascalImdb ;
dataOpts.eval_func = @pascal_eval_func ;
dataOpts.evalVersion = opts.evalVersion ;
dataOpts.displayResults = @displayPascalResults ;
dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;
dataOpts.imdbPath = fullfile(vl_rootnn, 'data/pascal/standard_imdb/imdb.mat') ;
dataOpts.configureImdbOpts = @configureImdbOpts ;
dataOpts.resultsFormat = 'minMax' ; 

% configure paths
tail = fullfile('evaluations', dataOpts.name, opts.modelName) ;
expDir = fullfile(vl_rootnn, 'data', tail) ;
resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
rawPredsFile = sprintf('%s-%s-raw-preds.mat', opts.modelName, opts.testset) ;
decodedPredsFile = sprintf('%s-%s-decoded.mat', opts.modelName, opts.testset) ;
evalCacheDir = fullfile(expDir, 'eval_cache') ;

if ~exist(evalCacheDir, 'dir') 
    mkdir(evalCacheDir) ;
    mkdir(fullfile(evalCacheDir, 'cache')) ;
end

cacheOpts.rawPredsCache = fullfile(evalCacheDir, rawPredsFile) ;
cacheOpts.decodedPredsCache = fullfile(evalCacheDir, decodedPredsFile) ;
cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
cacheOpts.evalCacheDir = evalCacheDir ;

% configure meta options
opts.dataOpts = dataOpts ;
opts.modelOpts = modelOpts ;
opts.batchOpts = batchOpts ;
opts.cacheOpts = cacheOpts ;

ssd_evaluation(expDir, net, opts) ;


% ------------------------------------------------------------------
function aps = pascal_eval_func(modelName, decodedPreds, imdb, opts)
% ------------------------------------------------------------------

numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
aps = zeros(numClasses, 1) ;

for c = 1:numClasses
    className = imdb.meta.classes{c + 1} ; % offset for background
    results = eval_voc(className, ...
                       decodedPreds.imageIds{c}, ...
                       decodedPreds.bboxes{c}, ...
                       decodedPreds.scores{c}, ...
                       opts.dataOpts.VOCopts, ...
                       'evalVersion', opts.dataOpts.evalVersion) ;
    fprintf('%s %.1\n', className, 100 * results.ap) ;
    aps(c) = results.ap ;
end
save(opts.cacheOpts.resultsCache, 'aps') ;

% -----------------------------------------------------------
function opts = configureImdbOpts(expDir, opts)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)

opts.dataOpts = configureVOC(expDir, opts.dataOpts, 'test') ;

%-----------------------------------------------------------
function dataOpts = configureVOC(expDir, dataOpts, testset) 
%-----------------------------------------------------------
% LOADPASCALOPTS Load the pascal VOC database options
%
% NOTE: The Pascal VOC dataset has a number of directories 
% and attributes. The paths to these directories are 
% set using the VOCdevkit code. The VOCdevkit initialization 
% code assumes it is being run from the devkit root folder, 
% so we make a note of our current directory, change to the 
% devkit root, initialize the pascal options and then change
% back into our original directory 

VOCRoot = fullfile(dataOpts.dataRoot, 'VOCdevkit2007') ;
VOCopts.devkitCode = fullfile(VOCRoot, 'VOCcode') ;

% check the existence of the required folders
assert(logical(exist(VOCRoot, 'dir')), 'VOC root directory not found') ;
assert(logical(exist(VOCopts.devkitCode, 'dir')), 'devkit code not found') ;

currentDir = pwd ; cd(VOCRoot) ; addpath(VOCopts.devkitCode) ;
VOCinit ; % VOCinit loads database options into a variable called VOCopts

dataDir = fullfile(VOCRoot, '2007') ;
VOCopts.localdir = fullfile(dataDir, 'local') ;
VOCopts.imgsetpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.imgpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.annopath = fullfile(dataDir, 'Annotations/%s.xml') ;
VOCopts.cacheDir = fullfile(expDir, '2007/Results/Cache') ;
VOCopts.drawAPCurve = false ;
VOCopts.testset = testset ;
detDir = fullfile(expDir, 'VOCdetections') ;

% create detection and cache directories if required
requiredDirs = {VOCopts.localdir, VOCopts.cacheDir, detDir} ;
for i = 1:numel(requiredDirs)
    reqDir = requiredDirs{i} ;
    if ~exist(reqDir, 'dir') , mkdir(reqDir) ; end
end

VOCopts.detrespath = fullfile(detDir, sprintf('%%s_det_%s_%%s.txt', 'test')) ;
dataOpts.VOCopts = VOCopts ;
cd(currentDir) ; % return to original directory

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------

fprintf('\n============\n') ;
fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
fprintf('\n============\n') ;
printPascalResults(opts.cacheOpts.evalCacheDir, 'orientation', 'portrait') ;
