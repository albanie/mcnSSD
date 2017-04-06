function ssd_train(expDir, opts, varargin) 

% ----------------------------------------------------------------
%                                                     Prepare imdb
% ----------------------------------------------------------------

if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
else
    imdb = opts.dataOpts.getImdb(opts) ;
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
end

[opts, imdb] = opts.dataOpts.prepareImdb(imdb, opts) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

if ~exist(expDir, 'dir')
    mkdir(expDir) ;
end

confirmConfig(expDir, opts) ;
net = opts.modelOpts.net_init(opts) ;

[net,info] = cnn_train_dag(net, imdb, ...
                    @(i,b) opts.modelOpts.get_batch(i, b, opts.batchOpts), ...
                    opts.train, ...
                    'expDir', expDir) ;

% --------------------------------------------------------------------
%                                                            Evaluatte
% --------------------------------------------------------------------

[net, modelName] = deployModel(expDir, opts) ;
opts.eval_func('net', net, 'modelName', modelName, 'gpus', opts.train.gpus) ;

% --------------------------------------------------
function [net, modelName] = deployModel(expDir, opts)
% --------------------------------------------------
bestEpoch = findBestEpoch(expDir, 'priorityMetric', 'mbox_loss', ...
                                                    'prune', true) ;
bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
deployPath = sprintf(opts.modelOpts.deployPath, bestEpoch) ;
opts.modelOpts.deploy_func(bestNet, deployPath, opts.modelOpts.numClasses) ;
net = dagnn.DagNN.loadobj(load(deployPath)) ;
[~,modelName,~] = fileparts(expDir) ; 
