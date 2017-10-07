function [nets,opts] = configureNets(opts)
% LOADNETS - load detection networks for multiscale evaluation
%    [NETS, OPTS] = LOADNETS(OPTS) loads networks into a consistent format for 
%    multiscale evaluation, This format stores each network in a cell 
%    array in autonn format. It also ensures that the BATCHOPTS is correctly
%    set up for the networks
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  if ~all(opts.msScales == 1) % multiscale eval
    msg = 'multiple nets should be supplied for multiscale evaluation' ;
    assert(numel(opts.net) >= 2, msg) ;
    stored = cellfun(@(x) {Layer.fromCompiledNet(x)}, opts.net) ;
    stored = cellfun(@(x) {setDetectionOpts(x, opts)}, stored) ;
    nets = cellfun(@(x) {Net(x)}, stored) ;
  else
    switch class(opts.net)
      case 'double' % handle empty input net case (opts.net = [])
        assert(isempty(opts.net), 'input net was not empty') ;
        dag = ssd_zoo(opts.modelName) ; 
        stored = Layer.fromDagNN(dag, @extras_autonn_custom_fn) ; 
      case 'dagnn.DagNN'
        stored = Layer.fromDagNN(opts.net, @extras_autonn_custom_fn) ; 
      case 'Net'
        stored = Layer.fromCompiledNet(opts.net) ;
      otherwise, error('unexpected net type:%s\n', class(opts.net)) ;
    end
    stored = setDetectionOpts(stored, opts) ;
    nets = {Net(stored{:})} ;
  end
        %nets = {opts.net} ;
    %if isempty(opts.net)
      %dag = ssd_zoo(opts.modelName) ; 
      %stored = Layer.fromDagNN(dag, @extras_autonn_custom_fn) ; 
      %stored = setDetectionOpts(stored, opts) ;
      %nets = {Net(stored{:})} ;
    %elseif isa(opts.net, 'dagnn.DagNN')
      %stored = Layer.fromDagNN(opts.net, @extras_autonn_custom_fn) ; 
      %nets = {Net(stored{:})} ;
    %elseif isa(opts.net, 'Net')
      %stored = Layer.fromCompiledNet(opts.net) ;
      %nets = {opts.net} ;
    %else
      %if ~all(opts.msScales == 1)
        %msg = 'multiple nets should be supplied for multiscale evaluation' ;
        %assert(numel(opts.net) >= 2, msg) ;
      %end
    %end

  % safety check here
  msg = ['SSD::Average training image (to be subtracted during inference)' ...
     ' must be supplied with the network to achieve reasonable performance'] ;
  assert(~isempty(nets{1}.meta.normalization.averageImage), msg) ;
  opts = configureBatchOpts(opts, nets) ;

% -------------------------------------------
function opts = configureBatchOpts(opts, nets)
% -------------------------------------------
% CONFIGUREBATCHOPTS(OPTS, NET) configures the batch options which are 
% network dependent.  Since multiple networks may be used in multiscale
% evaluation, batch options are set from the first network supplied in 
% the cell array of networks.

  first = nets{1} ;
  opts.batchOpts.imageSize = first.meta.normalization.imageSize ;
  opts.batchOpts.imMean = first.meta.normalization.averageImage ;
  if isfield(first.meta.normalization, 'scaleInputs')
    opts.batchOpts.scaleInputs = first.meta.normalization.scaleInputs ;
  else
    opts.batchOpts.scaleInputs = [] ;
  end

% ----------------------------------------------
function stored = setDetectionOpts(stored, opts)
% ----------------------------------------------
  detector = stored{1}.find('detection_out', 1) ;
  props = {'nmsThresh', 'keepTopK', 'nmsTopK', 'confThresh'} ;
  for ii = 1:numel(props)
    key = props{ii} ; value = opts.modelOpts.(key) ;
    detector.inputs = updateArgs(detector.inputs, key, value) ;
  end

% ------------------------------------
function x = updateArgs(x, key, value)
% ------------------------------------
%UPDATEARGS - update a cell array of options
%   X = UPDATEARGS(X, KEY, VALUE) updates a cell array X of inputs and key 
%   value pairs to include the pair given by { ... KEY, VALUE ...} if not 
%   already present, or overwrites it if it is already present.

  isKey = cellfun(@(y) isa(y, 'char'), x) ;
  present = contains(x(isKey), key) ;
  if ~any(present) % append k-v pair
    x(end+1:end+2) = {key, value} ;
  else % overwrite k-v pair
    keyIdx = find(isKey) ;
    x{keyIdx(present) + 1} = value ;
  end
