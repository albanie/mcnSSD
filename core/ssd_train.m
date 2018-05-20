function ssd_train(expDir, opts, varargin)
%SSD_TRAIN train an SSDnetwork end to end

  % Prepare imdb
  imdbPath = opts.dataOpts.imdbPath ;
  if exist(imdbPath, 'file')
      imdb = load(imdbPath) ;
  else
      imdb = opts.dataOpts.getImdb(opts) ;
      if ~exist(fileparts(imdbPath), 'dir'), mkdir(fileparts(imdbPath)) ; end
      save(imdbPath, '-struct', 'imdb') ;
  end

  [opts, imdb] = opts.dataOpts.prepareImdb(imdb, opts) ;

  % Train network
  if ~exist(expDir, 'dir'), mkdir(expDir) ; end
  confirmConfig(expDir, opts) ;
  net = opts.modelOpts.net_init(opts) ;
  [net,info] = cnn_train_autonn(net, imdb, ...
                      @(i,b) opts.modelOpts.get_batch(i, b, opts.batchOpts), ...
                      opts.train, ...
                      'expDir', expDir) ;


  % Evaluatte
  [net, modelName] = deployModel(expDir, opts) ;
  opts.eval_func('net', net, 'modelName', modelName, 'gpus', opts.train.gpus) ;

% ---------------------------------------------------
function [net, modelName] = deployModel(expDir, opts)
% ---------------------------------------------------
    bestEpoch = findBestEpoch(expDir, 'priorityMetric', opts.priorityMetric, ...
                                         'prune', opts.pruneCheckpoints) ;
    bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
    fprintf('best network %s\n', bestNet) ;
    deployPath = fullfile(expDir, 'deployed', ...
                     sprintf('local-%s-%s-%d-%d.mat', opts.modelOpts.type, ...
                     opts.dataOpts.name, opts.modelOpts.architecture, bestEpoch)) ;
    opts.modelOpts.deploy_func(bestNet, deployPath, opts.modelOpts.numClasses) ;
    storedNet = load(deployPath) ;
    net = Net(storedNet) ;
    [~,modelName,~] = fileparts(expDir) ;
