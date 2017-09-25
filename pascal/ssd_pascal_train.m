function ssd_pascal_train(varargin)

  opts.gpus = 1 ;
  opts.continue = true ;
  opts.confirmConfig = true ;
  opts.pruneCheckpoints = true ;
  opts.flipAugmentation = true ;
  opts.zoomAugmentation = false ;
  opts.patchAugmentation = true ;
  opts.distortAugmentation = true ;
  opts.useValForTraining = true ;
  opts.architecture = 300 ;
  opts.trainData = '0712' ;
  opts.use_vl_imreadjpeg = true ; 
  opts = vl_argparse(opts, varargin) ;

  % ---------------------------
  % configure training options
  % ---------------------------
  train.batchSize = 32 ;
  train.gpus = opts.gpus ;
  train.continue = opts.continue ;
  train.parameterServer.method = 'tmove' ;
  train.numSubBatches = ceil(4 / max(numel(train.gpus), 1)) ;

  % Since losses in each batch are computed as a function of ground truth
  % matches rather than batch size, we scale up the final derivative to "undo"
  % the derivative normalisation performed in cnn_train_dag
  derScale = train.batchSize / (train.numSubBatches * numel(opts.gpus)) ;
  train.derOutputs = {'mbox_loss', derScale} ;

  % -------------------------
  % configure dataset options
  % -------------------------
  dataOpts.name = 'pascal' ;
  dataOpts.trainData = opts.trainData ;
  dataOpts.testData = '07' ;
  dataOpts.flipAugmentation = opts.flipAugmentation ;
  dataOpts.zoomAugmentation = opts.zoomAugmentation ;
  dataOpts.patchAugmentation = opts.patchAugmentation ;
  dataOpts.useValForTraining = opts.useValForTraining ;
  dataOpts.distortAugmentation = opts.distortAugmentation ;
  dataOpts.zoomScale = 4 ;
  dataOpts.getImdb = @getPascalImdb ;
  dataOpts.prepareImdb = @prepareImdb ;
  dataOpts.dataRoot = fullfile(vl_rootnn, 'data', 'datasets') ;

  % -------------------------
  % configure model options
  % -------------------------
  modelOpts.type = 'ssd' ;
  modelOpts.numClasses = 21 ;
  modelOpts.clipPriors = false ;
  modelOpts.net_init = @ssd_init ;
  modelOpts.deploy_func = @ssd_deploy ;
  modelOpts.batchSize = train.batchSize ;
  modelOpts.get_batch = @ssd_train_get_batch ;
  modelOpts.architecture = opts.architecture ;


  % -------------------------
  %        Set learning rates
  % -------------------------
  warmup = 0.0001 ;
  steadyLR = 0.001 ;
  gentleLR = 0.0001 ;
  vGentleLR = 0.00001 ;

  if dataOpts.zoomAugmentation
      numSteadyEpochs = 155 ;
      numGentleEpochs = 38 ;
      numVeryGentleEpochs = 38 ;
  else
      numSteadyEpochs = 75 ;
      numGentleEpochs = 35 ;
      numVeryGentleEpochs = 0 ;
  end

  steady = steadyLR * ones(1, numSteadyEpochs) ;
  gentle = gentleLR * ones(1, numGentleEpochs) ;
  veryGentle = vGentleLR * ones(1, numVeryGentleEpochs) ;
  train.learningRate = [warmup steady gentle veryGentle] ;
  train.numEpochs = numel(train.learningRate) ;

  % ----------------------------
  % configure batch opts
  % ----------------------------
  batchOpts.clipTargets = true ;
  batchOpts.imageSize = repmat(modelOpts.architecture, 1, 2) ;
  batchOpts.patchOpts.use = dataOpts.patchAugmentation ;
  batchOpts.patchOpts.numTrials = 50 ;
  batchOpts.patchOpts.minPatchScale = 0.3 ;
  batchOpts.patchOpts.maxPatchScale = 1 ;
  batchOpts.patchOpts.minAspect = 0.5 ;
  batchOpts.patchOpts.maxAspect = 2 ;
  batchOpts.patchOpts.clipTargets = batchOpts.clipTargets ;

  batchOpts.flipOpts.use = dataOpts.flipAugmentation ;
  batchOpts.flipOpts.prob = 0.5 ;

  batchOpts.zoomOpts.use = dataOpts.zoomAugmentation ;
  batchOpts.zoomOpts.prob = 0.5 ;
  batchOpts.zoomOpts.minScale = 1 ;
  batchOpts.zoomOpts.maxScale = dataOpts.zoomScale ;

  batchOpts.distortOpts.use = dataOpts.distortAugmentation ;
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

  batchOpts.numThreads = 2 ;
  batchOpts.prefetch = false ; 
  batchOpts.useGpu = numel(train.gpus) >  0 ;
  batchOpts.use_vl_imreadjpeg = opts.use_vl_imreadjpeg ;
  batchOpts.resizeMethods = {'bilinear', 'box', 'nearest', 'bicubic', 'lanczos2'} ;


  % determine experiment name
  expName = getExpName(modelOpts, dataOpts) ;
  expDir = fullfile(vl_rootnn, 'data', dataOpts.name, expName) ;

  % -------------------------
  % configure paths
  % -------------------------
  dataOpts.imdbPath = fullfile(vl_rootnn, 'data', dataOpts.name, ...
                                              '/standard_imdb/imdb.mat') ;
  modelOpts.deployPath = fullfile(expDir, 'deployed', ...
                            sprintf('local-%s-%s-%d-%%d.mat', modelOpts.type, ...
                            dataOpts.name, modelOpts.architecture)) ;

  % -------------------------
  % configure meta options
  % -------------------------
  opts.train = train ;
  opts.dataOpts = dataOpts ;
  opts.modelOpts = modelOpts ;
  opts.batchOpts = batchOpts ;
  opts.eval_func = @ssd_pascal_evaluation ;

  % run
  ssd_train(expDir, opts) ;

% ---------------------------------------------------
function [opts, imdb] = prepareImdb(imdb, opts)
% ---------------------------------------------------

  % set path to VOC 2007 devkit directory 
  switch opts.dataOpts.trainData
      case '07'
          % to restrict to 2007, remove training 2012 data
          imdb.images.set(imdb.images.year == 2012 & imdb.images.set == 1) = -1 ;
      case '12'
          % to restrict to 2012, remove 2007 training data
          imdb.images.set(imdb.images.year == 2007 & imdb.images.set == 1) = -1 ;
      case '0712'
          ; % do nothing ( use full dataset) 
      otherwise
          error('Training data %s not recognized', opts.dataOpts.trainData) ;
  end

  opts.train.val = find(imdb.images.set == 2) ;

  if opts.dataOpts.useValForTraining
      opts.train.train = find(imdb.images.set == 2 | imdb.images.set == 1) ;
  end
