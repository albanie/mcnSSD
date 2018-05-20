function net = ssd_init(opts, varargin)
% SSD_INIT Initialize a Single Shot Multibox Detector Network
%   NET = SSD_INIT randomly initializes an SSD network architecture
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

% If set, this will force the layer pixel steps to exactly match
% SSD (useful as a sanity check, but not necessarily ideal for
% performance)

  opts.reproduceSSD = false ;
  opts = vl_argparse(opts, varargin) ;

  switch opts.modelOpts.architecture
    case 128, net = ssd_mini_init(opts) ; return ;
    case 300, numExtraConvs = 5 ; ns = 5 ;
    case 500, numExtraConvs = 5 ; ns = 6 ;
    case 512, numExtraConvs = 5 ; ns = 6 ;
    otherwise, error('arch %d unrecognised', opts.modelOpts.architecture) ;
  end

  % If the number of "source Layers" used for predictions has not been set,
  % then we use an appropriate number for each architecture
  if ~isfield(opts.modelOpts, 'numSources'), opts.modelOpts.numSources = ns ; end

  % load pre-trained base network (so far, only tested model is vgg-16-reduced)
  dag = ssd_zoo(opts.modelOpts.sourceModel) ;

  % modify trunk biases learnning rate and weight decay to match caffe
  params = {'conv1_1b','conv1_2b','conv2_1b','conv2_2b','conv3_1b', ...
            'conv3_2b','conv3_3b','conv4_1b','conv4_2b','conv4_3b' ...
            'conv5_1b','conv5_2b','conv5_3b','fc6b','fc7b'} ;
  for i = 1:length(params), dag = matchCaffeBiases(dag, params{i}) ; end

  % update fc6 to match SSD
  dag.layers(dag.getLayerIndex('fc6')).block.dilate = [6 6] ;
  dag.layers(dag.getLayerIndex('fc6')).block.pad = [6 6] ;

  % Truncate the layers following relu7 and rename input variable
  dag.removeLayer('fc8') ; dag.removeLayer('prob') ;
  dag.renameVar('x0', 'data') ;

  % Alter the pooling to match SSD
  dag.layers(dag.getLayerIndex('pool5')).block.stride = [1 1] ;
  dag.layers(dag.getLayerIndex('pool5')).block.poolSize = [3 3] ;
  dag.layers(dag.getLayerIndex('pool5')).block.pad = [1 1 1 1] ;

  % configure inputs
  data = Input('data') ;
  gtBoxes = Input('targets') ;
  gtLabels = Input('labels') ;

  % used by tukey and mAP layer
  epoch = Input('epoch') ;

  % additional input required by batch renormalization
  if opts.modelOpts.batchRenormalization
    clips = Input('clips') ;
    renormLR = {'learningRate', [2 1 opts.modelOpts.alpha]} ;
    opts.renormLR = renormLR ;
    opts.clips= clips;
  end

  % convert to autonn
  stored = Layer.fromDagNN(dag) ; net = stored{1} ;

  % for reproducibility, fix the seed
  rng(0) ;

  % add normalization layer
  base = net.find('relu4_3') ;
  weight = Param('value', ones(1,1,512, 'single') * 20, 'learningRate', 1) ;
  scaleNorm = Layer.create(@vl_nnscalenorm, {base{1}, weight}) ;
  scaleNorm.name = 'conv4_3_norm' ;

  % set scaleNorm and fc7_relu as the first and second "source layers"
  sourceLayers = {scaleNorm, net} ;

  % ------------------------------------------------------------
  %                                    add new conv layer stacks
  % ------------------------------------------------------------

  % Each additional conv stack is a form of "bottleneck" unit of
  % the form:
  %
  %     conv->relu->conv->relu
  %
  % Different conv stacks take different padding, stride and kernel
  % configurations. Each option is given as a 2 x n array, where the
  % first row of options are applied to the first conv in the stack,
  % and the second row of options is applied to the second row

  % padding configs
  pA = [ 0 0 0 0 ; 1 1 1 1 ] ;
  pB = [ 0 0 0 0 ; 0 0 0 0 ] ;

  % stride configs:
  sA = [ 1 1 ; 2 2 ] ;
  sB = [ 1 1 ; 1 1 ] ;

  % kernel sizes (e.g. kA implies 1x1 kernels followed by 3x3 kernels)
  kA = [1 ; 3] ;
  kB = [1 ; 4] ;

  % new layers
  stackOpts.prefix = {'conv6', 'conv7', 'conv8', 'conv9', 'conv10'} ;
  stackOpts.channelsIn =  {1024, 512, 256, 256, 256} ;
  stackOpts.bottlenecks = {256, 128, 128, 128, 128} ;
  stackOpts.channelsOut = {512, 256, 256, 256, 256} ;
  stackOpts.paddings = {pA, pA, pB, pB, pA} ;
  stackOpts.strides = {sA, sA, sB, sB, sB} ;
  stackOpts.ks = {kA, kA, kA, kA, kB} ;

  for i = 1:numExtraConvs
    net = add_conv_stack(net, i, stackOpts, opts) ;
    sourceLayers{end+1} = net ; %#ok
  end


  % --------------------------------------------------------------------
  %                                    compute multibox prior parameters
  % --------------------------------------------------------------------
  % select number of source layers and define the prior tiling.
  % The size of prior boxes are linearly spaced between each conv,
  % with the exception of the first set of priors, which is defined
  % separately (as "first")
  switch opts.modelOpts.architecture
    case 300
      minRatio = 20 ; maxRatio = 90 ; first = 10 ; % standard SSD-300
      aspectRatios = {2, [2, 3], [2, 3], [2, 3], 2, 2} ;
    case {512, 513}
      minRatio = 15 ; maxRatio = 90 ; first = 7 ; % standard SSD-512
      aspectRatios = {2, [2, 3], [2, 3], [2, 3], [2, 3], 2, 2} ;
    otherwise, error('unrecognised setup %s', opts.modelOpts.numSources) ;
  end

  modelOpts.numSources = numel(aspectRatios) ;
  sel = 1:modelOpts.numSources ;
  sourceLayers = sourceLayers(sel) ;

  % NOTE: These ratios are expressed as percentages
  priorOpts = getPriorOpts(sourceLayers, minRatio, maxRatio, first, opts) ;
  priorOpts.inputIm = data ;
  priorOpts.pixelStep = computePixelSteps(sourceLayers{end}, sourceLayers, opts) ;
  priorOpts.aspectRatios = aspectRatios(sel) ; % selected by source layer
  opts.priorOpts = priorOpts;

  predictors = cell(1, numel(sourceLayers)) ;
  for unit = 1:numel(sourceLayers)
    predictors{unit} = addMultiBoxLayers(unit, priorOpts, opts) ;
  end

  priorBoxLayers = cellfun(@(x) {x{1}}, predictors) ;
  dim = 1 ; fusedPriors = cat(dim, priorBoxLayers{:}) ;
  fusedPriors.name = 'mbox_priorbox' ;

  locLayers = cellfun(@(x) {x{2}}, predictors) ;
  confLayers = cellfun(@(x) {x{3}}, predictors) ; dim = 3 ;
  fusedLocs = cat(dim, locLayers{:}) ; fusedLocs.name = 'mbox_loc' ;
  fusedConfs = cat(dim, confLayers{:}) ; fusedConfs.name = 'mbox_conf' ;

  multiloss = add_loss(opts, gtBoxes, gtLabels, ...
                       fusedPriors, fusedConfs, fusedLocs) ;
  all_losses = {multiloss} ;

  % --------------------------------------------------------
  % Add detection subnetwork to generate mAP during training
  % --------------------------------------------------------
  % Note: this also means that at deployment time, we can simply
  % prune the training branches. However, it slows down training.
  if opts.track_map
    mAP = ssd_add_map(opts,fusedConfs,fusedLocs,fusedPriors,...
                                      gtLabels,gtBoxes,epoch,'');
   all_losses{end+1} = mAP ;
  end

  net = Net(all_losses{:}) ;
  if ~isempty(dag.meta.normalization.averageImage)
    net.meta.normalization.averageImage = dag.meta.normalization.averageImage ;
  else
    rgb = [122.771, 115.9465, 102.9801] ;
    net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;
  end
  net.meta.normalization.imageSize = repmat(opts.modelOpts.architecture, [1 2]) ;

% --------------------------------------------------------------------
function loss = add_loss(opts, gtBoxes, gtLabels, priors, confs, locs)
% --------------------------------------------------------------------
  numClasses = opts.modelOpts.numClasses ;
  overlapThreshold = opts.modelOpts.overlapThreshold ;

  % Matching and decoder layers
  args = {priors, gtBoxes, 'overlapThreshold', overlapThreshold} ;
  [matches, targets, tWeights, boxes] = Layer.create(@vl_nnmatchpriors, args, ...
                                                           'numInputDer', 0) ;
  matches.name = 'matches' ; targets.name = 'targets' ;
  tWeights.name = 'tWeights' ; boxes.name = 'boxes' ;

  % sample weighting
  % There are a couple of methods for doing sample weighting. THe first is
  % is to use a ranking loss to try to enable the use of all negatives in
  % training. The second is standard OHEM.
  lOpts = {'numClasses', numClasses, 'backgroundLabel', 1, ...
           'negPosRatio', opts.modelOpts.negPosRatio} ;
  args = [{confs, gtLabels, matches} lOpts] ;
  [hardNegs, exLabels, cWeights] = Layer.create(@vl_nnhardnegatives, ...
                                                   args, 'numInputDer', 0) ;
  hardNegs.name = 'hardNegs' ; cWeights.name = 'cWeights' ;
  exLabels.name = 'extendedLabels' ;

  args = {locs, confs, matches, hardNegs, 'numClasses', numClasses} ;
  largs = {'numInputDer', 2} ;
  [tarPreds, classPreds] = Layer.create(@vl_nnmultiboxcoder, args, largs{:}) ;
  tarPreds.name = 'mbox_loc' ; classPreds.name = 'mbox_conf' ;

  % Loss layers
  args = {classPreds, exLabels, 'instanceWeights', cWeights} ;
  softmaxlog = Layer.create(@vl_nnloss, args, 'numInputDer', 1) ;
  softmaxlog.name = 'conf_loss' ;

  args = {tarPreds, targets, 'instanceWeights', tWeights} ;
  regloss = Layer.create(@vl_nnhuberloss, args, 'numInputDer', 1) ;
  regloss.name = 'loc_loss' ;

  args = {softmaxlog, regloss, 'locWeight', opts.modelOpts.locWeight} ;
  loss = Layer.create(@vl_nnmultiboxloss, args) ;
  loss.name = 'mbox_loss' ;

% -----------------------------------------------------------------
function [confOut, locOut] = getOutSize(maxSize, aspectRatios, opts)
% -----------------------------------------------------------------
% THe filters must produce a prediction for each of the prior
% boxes associated with a source feature layer.  The number of
% prior boxes per feature is explained in detail in the SSD paper
% (essentially it is computed according to the number of specified
% aspect ratios)

  numBBoxOffsets = 4 ;
  priorsPerFeature = 1 + logical(maxSize) + numel(aspectRatios) * 2 ;
  confOut = opts.modelOpts.numClasses * priorsPerFeature ;
  locOut = numBBoxOffsets * priorsPerFeature ;

% ------------------------------------------------------------------------------
function priorOpts = getPriorOpts(sourceLayers, minRatio, maxRatio, first, opts)
% ------------------------------------------------------------------------------
  % minimum dimension of input image
  minDim = opts.modelOpts.architecture ;

  % set standard options
  priorOpts.kernelSize = [3 3] ;
  priorOpts.permuteOrder = [3 2 1 4] ;
  priorOpts.flattenAxis = 3 ;

  % Following the paper, the scale for conv4_3 is handled separately
  % when training on VOC0712, so we can compute the step as follows
  step = floor((maxRatio - minRatio) / max((numel(sourceLayers) - 2), 1)) ;
  effectiveMax = minRatio + (numel(sourceLayers) - 1) * step ;
  ratios = [first minRatio:step:maxRatio effectiveMax] ;
  for i = 1:numel(ratios) - 1
    priorOpts.minSizes(i) = minDim * ratios(i) / 100 ;
    priorOpts.maxSizes(i) = minDim * ratios(i + 1) / 100 ;
  end

  priorOpts.flip = true ;
  priorOpts.variance = [0.1 0.1 0.2 0.2]' ;
  priorOpts.sourceLayers = sourceLayers ;
  priorOpts.clip = opts.modelOpts.clipPriors ;

  % an offset is used to centre each prior box between
  % activations in the feature map (see Sec 2.2. of the
  % SSD paper)
  priorOpts.offset = 0.5 ;

  % Since the computed prior boxes are identical for a fixed size
  % input, we can cahce them after the first forward pass
  priorOpts.usePriorCaching = true ;

% ----------------------------------------------------
function net = add_conv_stack(net, i, stackOpts, opts)
% ----------------------------------------------------
  nonLin = true ; % add nonlinearity
  ks = stackOpts.ks{i}(1) ;
  stride = stackOpts.strides{i}(1,:) ;
  pad = stackOpts.paddings{i}(1,:) ;

  kernels = [ks ks stackOpts.channelsIn{i} stackOpts.bottlenecks{i}] ;
  name = sprintf('%s_1', stackOpts.prefix{i}) ;
  net = add_block(net, name, opts, kernels, nonLin, 'stride', stride, 'pad', pad) ;
  ks = stackOpts.ks{i}(2) ;
  stride = stackOpts.strides{i}(2,:) ;
  pad = stackOpts.paddings{i}(2,:) ;

  kernels = [ks ks stackOpts.bottlenecks{i} stackOpts.channelsOut{i}] ;
  name = sprintf('%s_2', stackOpts.prefix{i}) ;
  net = add_block(net, name, opts, kernels, nonLin,'stride',stride, 'pad', pad) ;

% ----------------------------------------------
function net = matchCaffeBiases(net, param)
% ----------------------------------------------
  % set the learning rate and weight decay of the
  % convolution biases to match caffe
  net.params(net.getParamIndex(param)).learningRate = 2 ;
  net.params(net.getParamIndex(param)).weightDecay = 0 ;

% ------------------------------------------------------------
function predictors = addMultiBoxLayers(unit, priorOpts, opts)
% ------------------------------------------------------------
  maxSize = priorOpts.maxSizes(unit) ;
  aspectRatios = priorOpts.aspectRatios{unit} ;
  [confOut, locOut] = getOutSize(maxSize, aspectRatios, opts) ;

  srcFeatures = priorOpts.sourceLayers{unit} ;
  priorBox = Layer.create(@vl_nnpriorbox, {srcFeatures, priorOpts.inputIm, ...
                          'aspectRatios', aspectRatios, ...
                          'pixelStep', priorOpts.pixelStep(unit), ...
                          'variance', priorOpts.variance, ...
                          'minSize', priorOpts.minSizes(unit), ...
                          'maxSize', priorOpts.maxSizes(unit), ...
                          'offset', priorOpts.offset, ...
                          'flip', priorOpts.flip, ...
                          'clip', priorOpts.clip}, ...
                          'numInputDer', 0) ;
  priorBox.name = sprintf('%s_priorbox', srcFeatures.name) ;

  % if the current layer is a pooling layer, we need to search back through
  % to the last conv layer to determine the current number of channels
  prev = srcFeatures.find(@vl_nnconv, -1) ;
  channelsIn = size(prev.inputs{3}.value, 1) ;
  assert(3 <= channelsIn && channelsIn <= 1024, 'unexpected num. of channels') ;

  % we do not include a non linear component in the prediction blocks
  nonLin = false ;
  ks = priorOpts.kernelSize(1:2) ; pad = 1 ;

  % add bbox regression predictors
  sz = [ks channelsIn locOut] ;
  name = sprintf('%s_loc', srcFeatures.name) ;
  loc = add_block(srcFeatures, name, opts, sz, nonLin, 'stride', 1, 'pad', pad) ;
  perm = Layer.create(@permute, {loc, priorOpts.permuteOrder}) ;
  perm.name = sprintf('%s_loc_perm', srcFeatures.name) ;
  flatLoc = Layer.create(@vl_nnflatten, {perm, priorOpts.flattenAxis}) ;
  flatLoc.name = sprintf('%s_loc_flat', srcFeatures.name) ;

  % add class confidence predictors
  sz = [ks channelsIn confOut] ;
  name = sprintf('%s_conf', srcFeatures.name) ;
  conf = add_block(srcFeatures, name, opts, sz, nonLin, 'stride', 1, 'pad', pad) ;
  perm = Layer.create(@permute, {conf, priorOpts.permuteOrder}) ;
  perm.name = sprintf('%s_conf_perm', srcFeatures.name) ;
  flatConf = Layer.create(@vl_nnflatten, {perm, priorOpts.flattenAxis}) ;
  flatConf.name = sprintf('%s_conf_flat', srcFeatures.name) ;
  predictors = { priorBox, flatLoc, flatConf } ;

% ---------------------------------------------------------------------
function net = add_block(net, name, opts, sz, nonLinearity, varargin)
% ---------------------------------------------------------------------

filters = Param('value', init_weight(sz, 'single'), 'learningRate', 1) ;
biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;

net = vl_nnconv(net, filters, biases, varargin{:}, ...
                'CudnnWorkspaceLimit', opts.modelOpts.CudnnWorkspaceLimit) ;
net.name = name ;

if nonLinearity
  bn = opts.modelOpts.batchNormalization ;
  rn = opts.modelOpts.batchRenormalization ;
  assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
  if bn
    net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
    net.name = sprintf('%s_bn', name) ;
  elseif rn
    net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ;
    net.name = sprintf('%s_rn', name) ;
  end
  net = vl_nnrelu(net) ;
  net.name = sprintf('%s_relu', name) ;
end



% -------------------------------------------------------------------------
function weights = init_weight(sz, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

  sc = sqrt(1/(sz(1)*sz(2)*sz(3))) ;
  weights = randn(sz, type)*sc ;

% -------------------------------------------------------------------------
function pixelSteps = computePixelSteps(net, sourceLayers, opts)
% -------------------------------------------------------------------------
  % compute the pixel steps
  trunk = Net(net, sourceLayers{1}) ; % include the scale norm layer
  imSz = opts.modelOpts.architecture ;
  inputs = {'data', zeros(imSz, imSz, 3, 1, 'single')} ;

  if opts.modelOpts.batchRenormalization
    clips = [1 0] ; inputs = [inputs, {'clips', clips}]  ;
  end

  trunk.eval(inputs, 'forward') ;

  pixelSteps = zeros(1, numel(sourceLayers)) ;
  for ii = 1:numel(sourceLayers)
    layer = sourceLayers{ii} ; feats = trunk.getValue(layer) ;
    featDim = size(feats, 1) ; pixelSteps(ii) = imSz / featDim ;
  end

  % In the original SSD implementation, these are mostly "squashed"
  % into powers of two.
  if opts.reproduceSSD
    switch opts.modelOpts.architecture
      case 300, pixelSteps = [8, 16, 32, 64, 100, 300] ;
      case 512, pixelSteps = [8, 16, 32, 64, 128, 256, 512] ;
      otherwise, error('disable `reproduceSSD` option for non standard archs') ;
    end
  end

