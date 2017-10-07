function results = ssd_coco_evaluation(varargin)
%SSD_COCO_EVALUATION Evaluate a trained SSD model on MS COCO
%
%   SSD_COCOL_EVALUATION(..'name', value) accepts the following 
%   options:
%
%   `gpus` :: 1
%    If provided, the gpu ids to be used for processing.
%
%   `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%    The path to the directory containing the coco data
%
%   `net` :: []
%    A cell array containing the `autonn` network object to be evaluated.  
%    If not supplied, a network will be loaded instead by name from the 
%    detector zoo. This can also be a cell array of multiple networks, to 
%    perform multiscale evaluation.
%
%   `modelName` :: 'ssd-mscoco-vggvd-512'
%    The name of the detector to be evaluated (used to generate output
%    file names, caches etc.)
%
%   `refreshCache` :: false
%    If true, overwrite previous predictions by any detector sharing the 
%    same model name, otherwise, load results directly from cache.
%
%   `useMiniVal` :: false
%    If true (and the testset is set to `val`), evaluate on the `mini-val` 
%    subsection of the coco data, rather than the full validation set.  This
%    setting is useful for evaluating models trained on coco-trainval135k.
%
%   `year` :: 2014
%    Select year of coco data to run evaluation on.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.net = [] ;
  opts.gpus = 1 ;
  opts.expDir = '' ; % preserve interface
  opts.useMiniVal = 1 ; 
  opts.testset = 'val' ;
  opts.visualise = false ;
  opts.refreshCache = false ;
  opts.modelName = 'ssd-mscoco-vggvd-300' ;

  % configure batch opts
  opts.batchOpts.batchSize = 8 ; 
  opts.batchOpts.numThreads = 4 ;
  opts.batchOpts.prefetch = true ;

  % configure dataset options
  opts.dataOpts.name = 'coco' ; 
  opts.dataOpts.year = 2014 ; 
  opts.dataOpts.scoreThresh = 0.01 ; 
  opts.dataOpts.decoder = 'parallel' ;
  opts.dataOpts.resultsFormat = 'minWH' ; 
  opts.dataOpts.getImdb = @getCocoImdb ; 
  opts.dataOpts.eval_func = @coco_eval_func ;
  opts.dataOpts.displayResults = @displayCocoResults ;
  opts.dataOpts.configureImdbOpts = @configureImdbOpts ;
  opts.dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;
  opts.dataOpts.labelMapFile = fullfile(vl_rootnn, 'data/coco/label_map.txt') ;
  opts.dataOpts.miniValPath = fullfile(opts.dataOpts.dataRoot, ...
    'mscoco/annotations/instances_minival2014.json') ; % only 2014 uses minival)

  % configure model options
  opts.modelOpts.keepTopK = 200 ;
  opts.modelOpts.predVar = 'detection_out' ;
  opts.modelOpts.get_eval_batch = @ssd_eval_get_batch ;

  % configure multiscale options
  opts.msOpts.use = false ;
  opts.msOpts.scales = [1 1.4] ;
  opts.msOpts.nmsThresh = 0.45 ;
  opts = vl_argparse(opts, varargin) ;

  [net, opts] = configureNets(opts) ; % configure network(s) for evaluation

  % cache configuration and model
  cacheOpts.refreshCache = opts.refreshCache ; 

  % select imdb based on year
  imdbName = sprintf('imdb%d.mat', opts.dataOpts.year) ;
  imdbOpts.conserveSpace = true ; imdbOpts.includeTest = true  ;
  opts.dataOpts.imdbPath = fullfile(vl_rootnn, 'data/coco/standard_imdb', imdbName) ;
  opts.imdbOpts = imdbOpts ; 

  % configure paths
  label = sprintf('%s%d', opts.dataOpts.name, opts.dataOpts.year) ;
  expDir = fullfile(vl_rootnn, 'data/evaluations', label, opts.modelName) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  rawPredsFile = sprintf('%s-%s-raw-preds.mat', opts.modelName, opts.testset) ;
  decodedPredsFile = sprintf('%s-%s-decoded.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, 'eval_cache') ;
  if ~exist(evalCacheDir, 'dir') 
    mkdir(evalCacheDir) ; mkdir(fullfile(evalCacheDir, 'cache')) ;
  end

  cacheOpts.rawPredsCache = fullfile(evalCacheDir, rawPredsFile) ;
  cacheOpts.decodedPredsCache = fullfile(evalCacheDir, decodedPredsFile) ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;
  opts.cacheOpts = cacheOpts ; 

  results = ssd_evaluation(expDir, net, opts) ;

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(~, opts, imdb)
% -----------------------------------------------------------
  % split images according to the popular "trainval35k" split commonly
  % used for ablation experiments
  switch opts.dataOpts.year
    case 2014
      if opts.useMiniVal
        annotations = gason(fileread(opts.dataOpts.miniValPath)) ;
        miniValIds = [annotations.images.id] ;
        fullValIms = find(imdb.images.set == 2) ;
        keep = ismember(imdb.images.id(fullValIms), miniValIds) ;
        imdb.images.set(fullValIms(~keep)) = 1 ;
      end
    case 2015 % do nothing
    case 2017 % do nothing
      % mini-check with val - debugging
      %fullValIms = find(imdb.images.set == 2) ;
      %imdb.images.set(fullValIms(100:end)) = -1 ;
  end

% ------------------------------------------------------------------
function aps = coco_eval_func(~, decoded, imdb, opts)
% ------------------------------------------------------------------
  aps = {} ; % maintain interface
  numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
  image_id = vertcat(decoded.imageIds{:}) ;
  category_id = arrayfun(@(x) {x*ones(1,numel(decoded.imageIds{x}))}, 1:numClasses) ;
  category_id = [category_id{:}]' ;

  labelMap = getCocoLabelMap('labelMapFile', opts.dataOpts.labelMapFile) ;
  category_id = arrayfun(@(x) labelMap(x), category_id) ;
  bbox = vertcat(decoded.bboxes{:}) ; score = vertcat(decoded.scores{:}) ;
  table_ = table(image_id, category_id, bbox, score) ; res = table2struct(table_) ;

  % encode as json (this may take a little while...., the gason func adds to storage)
  cocoJason = jsonencode(res) ;  template = 'detections_%s%d_%s%d.mat' ; 
  resFile = sprintf(template, opts.testset, opts.dataOpts.year, opts.modelName) ; 
  resPath = fullfile(opts.cacheOpts.evalCacheDir, resFile) ;
  fid = fopen(resPath, 'w') ; fprintf(fid, cocoJason) ; fclose(fid) ;
  fprintf('detection results have been saved to %s\n', resPath) ;

  if strcmp(opts.testset, 'val')
    %% initialize COCO ground truth api
    if opts.useMiniVal, mini = 'mini' ; else, mini = '' ; end 
    dataType = sprintf('%s%s%d', mini, opts.testset, opts.dataOpts.year) ;
    dataDir = fullfile(opts.dataOpts.dataRoot, 'mscoco') ;
    annFile = sprintf('%s/annotations/instances_%s.json',dataDir,dataType) ;
    cocoGt = CocoApi(annFile) ; % load ground truth
    cocoDt = cocoGt.loadRes(resPath) ; % load detections
    cocoEval = CocoEval(cocoGt, cocoDt, 'bbox') ;
    imgIds = sort(cocoGt.getImgIds());  cocoEval.params.imgIds = imgIds ;
    cocoEval.evaluate() ; cocoEval.accumulate() ; cocoEval.summarize() ;
    aps = cocoEval.eval ;
    if opts.visualise, visualizeRes(res, cocoEval, imdb) ; end % for debugging
  end

% ------------------------------------------------------------------------
function visualizeRes(res, cocoEval,imdb)
% ------------------------------------------------------------------------
  res = res([res.score] > 0.6) ; % restrict to confident preds
  sampleSize = 100 ; sample = randi(numel(res), 1, sampleSize) ;
  [revLabelMap,labels] = getCocoLabelMap('reverse', true) ;
  for ii = 1:numel(sample)
    rec = res(sample(ii)) ;
    id = find(imdb.images.id == rec.image_id) ;
    imName = imdb.images.name{id} ; template = imdb.images.paths{id} ;
    imPath = sprintf(template, imName) ; im = single(imread(imPath)) ;
    label = labels{revLabelMap(rec.category_id)} ;
    box = rec.bbox ; score = rec.score ;  
    gtIds = [cocoEval.cocoGt.data.annotations.image_id] ;
    gt = cocoEval.cocoGt.data.annotations((gtIds == rec.image_id)) ;
    drawCocoBoxes(im, box, score, label, 'format', 'MinWH', 'gt', gt) ;
  end

% ---------------------------------------------------------------------------
function displayCocoResults(~, aps, opts)
% ---------------------------------------------------------------------------
  if ~strcmp(opts.testset, 'val'), return ; end 
   a = 1 ; m = 3 ; % max dets = 100
  s = aps.precision(:,:,:,a,m) ; cocoScore = mean(s(s>=0)) * 100 ;
  fprintf('coco score for %s: %g \n', opts.modelName, cocoScore) ;
