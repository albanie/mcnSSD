function fix_ssd_imports(varargin)
%FIX_SSD_IMPORTS - clean up imported caffe models
%   FIX_SSD_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions. 
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.numClasses = 21 ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  % select model
  res = dir(fullfile(opts.modelDir, '*.mat')) ; modelNames = {res.name} ;
  modelNames = modelNames(contains(modelNames, 'ssd')) ;

  for mm = 1:numel(modelNames)
    modelPath = fullfile(opts.modelDir, modelNames{mm}) ;
    fprintf('fixing name scheme for %s\n', modelNames{mm}) ;
    net = load(modelPath) ; 

    % fix naming convention
    for ii = 1:numel(net.layers)
      net.layers(ii).name = strrep(net.layers(ii).name, '/', '_') ;
      net.layers(ii).inputs = strrep(net.layers(ii).inputs, '/', '_') ;
      net.layers(ii).outputs = strrep(net.layers(ii).outputs, '/', '_') ;
      net.layers(ii).params = strrep(net.layers(ii).params, '/', '_') ;
    end
    for ii = 1:numel(net.params)
      net.params(ii).name = strrep(net.params(ii).name, '/', '_') ;
    end

    for ii = 1:numel(net.layers)
      % fix priorboxes
      if strcmp(net.layers(ii).type, 'dagnn.PriorBox')
        pixelStep = net.layers(ii).block.pixelStep ;
        if isempty(pixelStep) || pixelStep == 1
          net.layers(ii).block.pixelStep = 0 ;
        end
        if isempty(net.layers(ii).block.offset)
          net.layers(ii).block.offset = 0.5 ;
        end
        if isempty(net.layers(ii).block.maxSize)
          net.layers(ii).block.maxSize = 0 ;
        end
      end

      % reverse dims for faster detection on final set
      if strcmp(net.layers(ii).name, 'mbox_priorbox')
        net.layers(ii).block.dim = 1 ;
      end

      % switch to softmax transpose
      if strcmp(net.layers(ii).type, 'dagnn.SoftMax')
        net.layers(ii).type = 'dagnn.SoftMaxTranspose' ;
        net.layers(ii).block = struct('dim', 1) ;
      end

      % switch to softmax transpose
      if strcmp(net.layers(ii).type, 'dagnn.Reshape')
        shape = net.layers(ii).block.shape ;
        if isequal(shape, [0, -1, 21])
          net.layers(ii).block.shape = {opts.numClasses  []  1} ;
        end
      end
    end

    % fix meta 
    fprintf('adding info to %s (%d/%d)\n', modelPath, mm, numel(modelNames)) ;
    if contains(modelPath, '300')
      net.meta.normalization.imageSize = [300 300 3] ;
    elseif contains(modelPath, '512')
      net.meta.normalization.imageSize = [512 512 3] ;
    else
      keyboard
    end
    if ~contains(modelPath, 'mobilenet')
      net.meta.normalization.averageImage = [123 117 104] ; % caffe pretraining
      net.meta.normalization.scaleInputs = [] ; % google pretraining
    else
      net.meta.normalization.averageImage = [127.5, 127.5, 127.5] ; 
      net.meta.normalization.scaleInputs = 0.007843 ; % google pretraining
    end
    net = dagnn.DagNN.loadobj(net) ; 
    net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
  end
