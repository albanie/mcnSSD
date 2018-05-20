function ssd_pascal_train(varargin)
%SSD_PASCAL_TRAIN Train an SSD detector on pascal VOC data
%   SSD_PASCAL_TRAIN performs a full training run of SSD detector on the
%   Pascal VOC dataset. A number of options and settings are
%   provided for training.  The defaults should reproduce the experiment
%   described in the original SSD paper (linked in README.md).
%
%   SSD_PASCAL_TRAIN(..'name', value) accepts the following options:
%
%   `confirmConfig` :: true
%    Ask for confirmation of the experimental settings before running the
%    experiment
%
%   `pruneCheckpoints` :: true
%    Determines whether intermediate training files should be cleared to save
%    space after the training run has completed.
%
%   `priorityMetric`:: 'mbox_loss'
%    Determines the metric by which to rank the performance of the saved
%    checkpoints.
%
% ----------------------------------------------------------------------
%   `train` :: struct(...)
%    A structure of options for training, with the following fields:
%
%      `gpus` :: 1
%       If provided, the gpu ids to be used for processing.
%
%      `batchSize` :: 32
%       Number of images per batch during training.
%
%      `continue` :: true
%       Resume training from previous checkpoint.
%
% ----------------------------------------------------------------------
%   `modelOpts` :: struct(...)
%    A structure of options for the model, with the following fields:
%
%      `architecture` :: 300
%       The SSD architecture to train (the number denotes input image size)
%
%      `numClasses` :: 21
%       The number of class specific predictors to include in the network
%       architecture (this includes a background class)
%
%      `clipPriors` :: false
%       Clip SSD priors to lie completely within the bounds of the input
%       image.
%
%      `sourceModel` :: 'vgg-vd-16-reduced'
%       The name of the feature extractor used as a trunk for SSD (by default
%       the atrous version of vgg-vd-16 is used).
%
%      `overlapThreshold` :: 0.5
%       The threshold used to determine whether a ground truth annotation is
%       to be matched to a given prior box (the prior box then becomes a
%       positive example during training).
%
%      `negPosRatio` :: 3
%       The ratio of negative-to-positive samples used during training.
%
%      `locWeight` :: 1
%       A scalar which weights the loss contribution of the regression loss
%       against the class confidence loss.
%
% ----------------------------------------------------------------------
%   `dataOpts` :: struct(...)
%    A structure of options for the data, with the following fields:
%
%      `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%       The path to the directory containing the Pascal VOC data data
%
%      `useValForTraining` :: true
%       Whether the validation set (as defined in the original challenge)
%       should be included in the training set.
%
%      `zoomScale` :: 4
%       Zoom magnitude used by SSD-style data augmentation
%
%      `flipAugmentation` :: true
%       Whether flipped images should be used in the training procedure.
%
%      `zoomAugmentation` :: false
%       Use "zoom" augmentation to improve performance (but longer training)
%
%      `patchAugmentation` :: true
%       Use SSD-style "patch" augmentation to improve performance
%
%      `distortAugmentation` :: false
%       Use SSD-style "distortion" augmentation to improve performance
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.track_map = false ;
  opts.confirmConfig = true ;
  opts.pruneCheckpoints = false ;
  opts.priorityMetric = 'mbox_loss' ;

  % configure training options
  opts.train.gpus = 1 ;
  opts.train.batchSize = 32 ;
  opts.train.continue = true ;
  opts.train.parameterServer.method = 'tmove' ;
  opts.train.stats = {'conf_loss', 'loc_loss', 'mbox_loss'} ;
  opts.train.extractStatsFn = @extractUnnormalizedStats ;

  % configure model options
  opts.modelOpts.type = 'ssd' ;
  opts.modelOpts.locWeight = 1 ;
  opts.modelOpts.numClasses = 21 ;
  opts.modelOpts.negPosRatio = 3 ;
  opts.modelOpts.architecture = 300 ;
  opts.modelOpts.clipPriors = false ;
  opts.modelOpts.net_init = @ssd_init ;
  opts.modelOpts.overlapThreshold = 0.5 ;
  opts.modelOpts.deploy_func = @ssd_deploy ;
  opts.modelOpts.get_batch = @ssd_train_get_batch ;
  opts.modelOpts.batchNormalization = false ;
  opts.modelOpts.batchRenormalization = false ;
  opts.modelOpts.CudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
  opts.modelOpts.sourceModel = 'vgg-vd-16-reduced' ;

  % configure dataset options
  opts.dataOpts.name = 'pascal' ;
  opts.dataOpts.trainData = '0712' ;
  opts.dataOpts.testData = '07' ;
  opts.dataOpts.flipAugmentation = true ;
  opts.dataOpts.zoomAugmentation = false ;
  opts.dataOpts.patchAugmentation = true ;
  opts.dataOpts.useValForTraining = true ;
  opts.dataOpts.distortAugmentation = true ;
  opts.dataOpts.getImdb = @getCombinedPascalImdb ;
  opts.dataOpts.prepareImdb = @prepareImdb ;
  opts.dataOpts.zoomScale = 4 ;
  opts.dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;
  opts = vl_argparse(opts, varargin) ;

  % Since losses in each batch are computed as a function of ground truth
  % matches rather than batch size, we scale up the final derivative to "undo"
  % derivative normalisation
  opts.train.numSubBatches = ceil(4 / max(numel(opts.train.gpus), 1)) ;
  scaleFactor = opts.train.numSubBatches * numel(opts.train.gpus) ;
  derScale = opts.train.batchSize / scaleFactor ;
  opts.train.derOutputs = derScale ;

  % Set learning rates to match caffe implementation
  steadyLR = 0.001 ; gentleLR = 0.0001 ; vGentleLR = 0.00001 ;
  if opts.dataOpts.zoomAugmentation
    numSteadyEpochs = 155 ; numGentleEpochs = 38 ; numVeryGentleEpochs = 38 ;
  else
    numSteadyEpochs = 75 ; numGentleEpochs = 35 ; numVeryGentleEpochs = 0 ;
  end

  steady = steadyLR * ones(1, numSteadyEpochs) ;
  gentle = gentleLR * ones(1, numGentleEpochs) ;
  veryGentle = vGentleLR * ones(1, numVeryGentleEpochs) ;
  opts.train.learningRate = [steady gentle veryGentle] ;
  opts.train.numEpochs = numel(opts.train.learningRate) ;
  opts.modelOpts.batchSize = opts.train.batchSize ;

  % Configure batch opts. NOTE: The SSD training process uses a variety of data
  % augmentation techiques, the settings listed below are designed to
  % reproduce the exeriments in the original paper.
  batchOpts.numThreads = 2 ;
  batchOpts.prefetch = true ;
  batchOpts.use_vl_imreadjpeg = true ;
  batchOpts.clipTargets = true ;
  batchOpts.imageSize = repmat(opts.modelOpts.architecture, 1, 2) ;
  batchOpts.patchOpts.use = opts.dataOpts.patchAugmentation ;
  batchOpts.patchOpts.numTrials = 50 ;
  batchOpts.patchOpts.minPatchScale = 0.3 ;
  batchOpts.patchOpts.maxPatchScale = 1 ;
  batchOpts.patchOpts.minAspect = 0.5 ;
  batchOpts.patchOpts.maxAspect = 2 ;
  batchOpts.patchOpts.clipTargets = batchOpts.clipTargets ;
  batchOpts.flipOpts.use = opts.dataOpts.flipAugmentation ;
  batchOpts.flipOpts.prob = 0.5 ;
  batchOpts.zoomOpts.use = opts.dataOpts.zoomAugmentation ;
  batchOpts.zoomOpts.prob = 0.5 ;
  batchOpts.zoomOpts.minScale = 1 ;
  batchOpts.zoomOpts.maxScale = opts.dataOpts.zoomScale ;
  batchOpts.distortOpts.use = opts.dataOpts.distortAugmentation ;
  batchOpts.distortOpts.brightnessProb = 0.5 ;
  batchOpts.distortOpts.contrastProb = 0.5 ;
  batchOpts.distortOpts.saturationProb = 0.5 ;
  batchOpts.distortOpts.hueProb = 0.5 ;
  batchOpts.distortOpts.brightnessDelta = 32 ;
  batchOpts.distortOpts.contrastLower = 0.5 ;
  batchOpts.distortOpts.contrastUpper = 1.5 ;
  batchOpts.distortOpts.hueDelta = 18 ;
  batchOpts.distortOpts.saturationLower = 0.5 ;
  batchOpts.distortOpts.saturationUpper = 1.5 ;
  batchOpts.distortOpts.randomOrderProb = 0 ;
  batchOpts.useGpu = numel(opts.train.gpus) >  0 ;
  batchOpts.resizeMethods = {'bilinear', 'box', 'nearest', 'bicubic', 'lanczos2'} ;

  % determine experiment name
  expName = getExpName(opts.modelOpts, opts.dataOpts) ;
  expDir = fullfile(vl_rootnn, 'data', opts.dataOpts.name, expName) ;

  % configure paths
  base = fullfile(vl_rootnn, 'data', opts.dataOpts.name) ;
  opts.dataOpts.imdbPath = fullfile(base, 'standard_imdb/imdb.mat') ;
  opts.modelOpts.deployPath = fullfile(expDir, 'deployed', ...
                   sprintf('local-%s-%s-%d-%%d.mat', opts.modelOpts.type, ...
                   opts.dataOpts.name, opts.modelOpts.architecture)) ;

  % configure meta options
  opts.batchOpts = batchOpts ;
  opts.eval_func = @ssd_pascal_evaluation ;

  ssd_train(expDir, opts) ;

% ---------------------------------------------------
function [opts, imdb] = prepareImdb(imdb, opts)
% ---------------------------------------------------
  % set path to VOC 2007 devkit directory
  switch opts.dataOpts.trainData
    case '07' % to restrict to 2007, remove training 2012 data
      imdb.images.set(imdb.images.year == 2012) = -1 ;
    case '12' % to restrict to 2012, remove 2007 training data
      imdb.images.set(imdb.images.year == 2007) = -1 ;
    case '0712' % do nothing ( use full dataset)
    otherwise, error('Data %s unrecognized', opts.dataOpts.trainData) ;
  end

  opts.train.val = find(imdb.images.set == 2) ;
  if opts.dataOpts.useValForTraining
    opts.train.train = find(imdb.images.set == 2 | imdb.images.set == 1) ;
  end

% --------------------------------------------------------------------------------
function stats = extractUnnormalizedStats(stats, net, sel, batchSize, normalized)
% --------------------------------------------------------------------------------
  for i = 1:numel(sel)
    name = net.forward(sel(i)).name ;
    if ~isfield(stats, name), stats.(name) = 0 ; end
    newValue = gather(sum(net.vars{net.forward(sel(i)).outputVar(1)}(:))) ;

    % Update running average (same work as dagnn.Loss), but without normalizing
    % by batch size. NOTE: This means that the final iteration average can be
    % slightly inaccurate (but provides a useful approximation of the loss)
    iter = floor(stats.num / batchSize) ;
    stats.(name) = ((iter - 1) * stats.(name) + newValue) / iter ;
  end
