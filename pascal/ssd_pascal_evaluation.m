function ssd_pascal_evaluation(varargin)
%SSD_PASCAL_EVALUATION evaluate SSD detector on pascal VOC
%   SSD_PASCAL_EVALUATION computes and evaluates a set of detections
%   for a given SSD detector on the Pascal VOC dataset.
%
%   SSD_PASCAL_EVALUATION(..'name', value) accepts the following 
%   options:
%
%   `testset` :: 'test'
%    The subset of pascal VOC to be used for evaluation.
%
%   `year` :: 2007
%    The year of the challenge to evalutate on. Currently 2007 (val and test)
%    and 2012 (val) are supported.  Predictions for 2012 test must be submitted
%    to the official evaluation server to obtain scores.
%
%   `gpus` :: []
%    If provided, the gpu ids to be used for processing.
%
%   `evalVersion` :: 'fast'
%    The type of VOC evaluation code to be run.  The options are 'official', 
%    which runs the original (slow) pascal evaluation code, or 'fast', which
%    runs an optimised version which is useful during development.
%
%   `nms` :: 'cpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `modelName` :: 'faster-rcnn-vggvd-pascal'
%    The name of the detector to be evaluated (used to generate output
%    file names, caches etc.)
%
%   `refreshCache` :: false
%    If true, overwrite previous predictions by any detector sharing the 
%    same model name, otherwise, load results directly from cache.
%
%   `net` :: []
%    A cell array containing the `autonn` network object to be evaluated.  
%    If not supplied, a network will be loaded instead by name from the 
%    detector zoo. This can also be a cell array of multiple networks, to 
%    perform multiscale evaluation.  Multiple networks are required for 
%    multiscale evalaution because the SSD architecture does not perform well
%    if applied naively at different scales.  
%
% ----------------------------------------------------------------------------
%   `modelOpts` :: struct(...)
%    A structure of options relating to the properties of the model, with the 
%    following fields:
%      `predVar` :: 'detection_out'
%       The name of the output prediction variable of the network 
%
%      `scales` :: [0.8 1 1.4] 
%       The input size scales that should be combined at test time
%
%      `keepTopK` :: 200 
%       The number of predictions that should be kept for evaluation at test
%       time. Note: if this is significantly increased, inference is slower 
%       and memory issues can occur.
%
%      `nmsTopK` :: 400 
%       The number of intermediate predictions that should be kept for the 
%       final stage of non maximum supression. Note: if this is significantly 
%       increased, inference is slower and memory issues can occur.
%
%      `confThresh` :: 0.01 
%       The minimum confidence required for a prediction to be scored as a 
%       "detection" by the network.
%
%      `nmsThresh` :: 0.45 
%       The NMS threshold used to select predictions on a single image.
%
%      `outCols` :: 6
%       The number of columns forming the structured output of the detector.
%       By default this is 6 (where each row consists of a class label, a 
%       confidence and 4 values describing the assoicated bounding box).
%
% ----------------------------------------------------------------------------
%   `msOpts` :: struct(...)
%    A structure of options prescribing the multiscale evaluation settings with
%    the following fields:
%      `use` :: false 
%       Whether or not to perform multiscale evaluation
%
%      `scales` :: [0.8 1 1.4] 
%       The input size scales that should be combined at test time
%
%      `nmsThresh` :: 0.45 
%       The NMS threshold used to combine predictions from multiple scales
%
%   `dataOpts` :: struct(...)
%    A structure of options setting paths for the data
%
%   `batchOpts` :: struct(...)
%    A structure of options relating to the properties of each batch of images.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.net = [] ;
  opts.gpus = 3 ;
  opts.year = 2007 ;
  opts.msScales = 1 ; % by default, only single scale selection is used
  opts.testset = 'test' ; 
  opts.evalVersion = 'fast' ;
  opts.refreshCache = true ;
  opts.modelName = 'ssd-pascal-vggvd-300' ;

  % configure batch opts
  opts.batchOpts.batchSize = 8 ;
  opts.batchOpts.numThreads = 4 ;
  opts.batchOpts.prefetch = true ;

  % configure dataset options
  opts.dataOpts.name = 'pascal' ;
  opts.dataOpts.decoder = 'serial' ;
  opts.dataOpts.customDecoder = [] ;
  opts.dataOpts.getImdb = @getCombinedPascalImdb ;
  opts.dataOpts.resultsFormat = 'minMax' ; 
  opts.dataOpts.eval_func = @pascal_eval_func ;
  opts.dataOpts.evalVersion = opts.evalVersion ;
  opts.dataOpts.displayResults = @displayPascalResults ;
  opts.dataOpts.configureImdbOpts = @configureImdbOpts ;
  opts.dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;
  opts.dataOpts.imdbPath = fullfile(vl_rootnn, ...
                                       'data/pascal/standard_imdb/imdb.mat') ;
  % configure model options
  opts.modelOpts.nmsTopK = 400 ;
  opts.modelOpts.keepTopK = 200 ;
  opts.modelOpts.nmsThresh = 0.45 ;
  opts.modelOpts.confThresh = 0.01 ;
  opts.modelOpts.outCols = 6 ;
  opts.modelOpts.predVar = 'detection_out' ;
  opts.modelOpts.get_eval_batch = @ssd_eval_get_batch ;

  % configure multiscale options
  opts.msOpts.use = false ;
  opts.msOpts.scales = [1 1.4] ;
  opts.msOpts.nmsThresh = 0.45 ;
  opts = vl_argparse(opts, varargin) ;

  [net, opts] = configureNets(opts) ; % configure network(s) for evaluation

  % configure paths
  tail = fullfile('evaluations', opts.dataOpts.name, opts.modelName) ;
  expDir = fullfile(vl_rootnn, 'data', tail) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  rawPredsFile = sprintf('%s-%s-raw-preds.mat', opts.modelName, opts.testset) ;
  decodedPredsFile = sprintf('%s-%s-decoded.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, sprintf('eval_cache-%d', opts.year)) ;
  if ~exist(evalCacheDir, 'dir') 
    mkdir(evalCacheDir) ; mkdir(fullfile(evalCacheDir, 'cache')) ;
  end

  % cache configuration 
  cacheOpts.rawPredsCache = fullfile(evalCacheDir, rawPredsFile) ;
  cacheOpts.decodedPredsCache = fullfile(evalCacheDir, decodedPredsFile) ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;
  cacheOpts.refreshCache = opts.refreshCache ;
  opts.cacheOpts = cacheOpts ;
  ssd_evaluation(expDir, net, opts) ;

% ------------------------------------------------------------------
function aps = pascal_eval_func(modelName, decodedPreds, imdb, opts)
% ------------------------------------------------------------------
  fprintf('evaluating predictions for %s\n', modelName) ;
  numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
  aps = zeros(numClasses, 1) ;
  if (opts.year == 2012) && strcmp(opts.testset, 'test')
    fprintf('preds on 2012 test set must be submitted to the eval server\n') ;
    keyboard % TODO(samuel): Add support for output format
  else
    for c = 1:numClasses
      className = imdb.meta.classes{c + 1} ; % offset for background
      results = eval_voc(className, ...
                         decodedPreds.imageIds{c}, ...
                         decodedPreds.bboxes{c}, ...
                         decodedPreds.scores{c}, ...
                         opts.dataOpts.VOCopts, ...
                         'evalVersion', opts.dataOpts.evalVersion, ...
                         'year', opts.year) ;
      fprintf('%s %.1\n', className, 100 * results.ap) ;
      aps(c) = results.ap_auc ;
    end
    save(opts.cacheOpts.resultsCache, 'aps') ;
  end

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(expDir, opts, imdb)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)
  switch opts.year   
    case 2007, imdb.images.set(imdb.images.year == 2012) = -1 ;   
    case 2012, imdb.images.set(imdb.images.year == 2007) = -1 ;   
    case 0712 % do nothing    
    otherwise, error('Data from year %s not recognized', opts.year) ;    
  end   
  % ignore images that do not reside in the classification & detection challenge
  imdb.images.set(~imdb.images.classification) = -1 ;
  VOCopts = configureVOC(expDir, opts.dataOpts.dataRoot, opts.year) ;
  VOCopts.testset = opts.testset ;
  opts.dataOpts.VOCopts = VOCopts ;
  %testIms = (imdb.images.set == 3) ;
  %imdb.images.set(testIms(20:end)) = - 1;

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------

fprintf('\n============\n') ;
fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
fprintf('\n============\n') ;
