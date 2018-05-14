function net = ssd_mini_init(opts)
% SSD_MINI_INIT Initialize a miniature Single Shot Multibox Detector Network
%   NET = SSD_MINI_INIT randomly initializes a lightweight SSD network, 
%   designed for rapid prototyping.  This should not be considered stable
%   or reliable code.

  % for reproducibility, fix the seed
  rng(0, 'twister') ;

  data = Input('data') ;
  gtBoxes = Input('gtBoxes') ;
  gtLabels = Input('gtLabels') ; 
  numClasses = Input('numClasses') ;

  % used by tukey and mAP layer
  epoch = Input('epoch') ;

  if opts.modelOpts.batchRenormalization
    clips = Input('clips') ; 
    renormLR = {'learningRate', [2 1 opts.modelOpts.alpha]} ;
    opts.renormLR = renormLR ;
    opts.clips= clips;
  end


% insert data augmentaiton layer
%augData = Layer.create(@vl_nnaugment, {data, augs, gtBoxes, gtLabels}, 'numInputDer', 0) ;

if opts.modelOpts.pretrained
  path = fullfile(vl_rootnn, 'data/models-import/vgg-vd-16-reduced.mat') ;
  nn = load(path) ;
  dag = dagnn.DagNN.fromSimpleNN(nn) ;
  dag.layers(dag.getLayerIndex('fc6')).block.dilate = [6 6] ;
  dag.layers(dag.getLayerIndex('fc6')).block.pad = [6 6] ;
  dag.removeLayer('prob') ; 
  dag.removeLayer('fc8') ;

  tmp = Layer.fromDagNN(dag) ;
  r7 = tmp{1} ;
  c7 = r7.inputs{1} ;
  r6 = c7.inputs{1} ;

  % retrieve references
  r5 = r6.inputs{1}.find(@vl_nnrelu, -1) ;
  p4 = r5.find(@vl_nnpool, -1) ;
  p3 = p4.inputs{1}.find(@vl_nnpool, -1) ;

  if 0 
    imSz = opts.architecture ;
    inputs = {'x0', randn(imSz, imSz, 3, 1, 'single')} ;
    trunk = Net(c7) ;
    trunk.eval(inputs, 'forward') ;
  end

  % rename input from x0 to data
  l = p3 ;
  while ~strcmp(l.name, 'x0')
    l = l.inputs{1} ;
  end
  l.name = 'data' ;

else
  nonLin = true ; % add nonlinearity
  switch opts.modelOpts.archName
    case 'A'
      r1 = add_block(data, opts, [3, 3, 3, 96], nonLin, 'stride', 1, 'pad', 1) ;
      p1 = vl_nnpool(r1, [3 3], 'stride', 2, 'pad', 1) ;
      r2 = add_block(p1, opts, [3, 3, 96, 256], nonLin, 'stride', 1, 'pad', 1) ;
      p2 = vl_nnpool(r2, [3 3], 'stride', 2, 'pad', 1) ;
      r3 = add_block(p2, opts, [3, 3, 256, 256], nonLin, 'stride', 1, 'pad', 1) ;
      p3 = vl_nnpool(r3, [3 3], 'stride', 2, 'pad', 1) ;
      r4 = add_block(p3, opts, [3, 3, 256, 256], nonLin, 'stride', 1, 'pad', 1) ;
      p4 = vl_nnpool(r4, [3 3], 'stride', 2, 'pad', 0) ;
      r5 = add_block(p4, opts, [3, 3, 256, 512], nonLin, 'stride', 1, 'pad', 0) ;
      r6 = add_block(r5, opts, [3, 3, 512, 512], nonLin, 'stride', 1, 'pad', 0) ;

      sourceLayers = { p3, p4, r5, r6 } ;  
    case 'B'
      r1 = add_block(data, opts, [3, 3, 3, 96], nonLin, 'stride', 1, 'pad', 1) ;
      r3 = add_block(r1, opts, [3, 3, 96, 128], nonLin, 'stride', 1, 'pad', 1) ;
      p1 = vl_nnpool(r3, [3 3], 'stride', 2, 'pad', 1) ;
      r4 = add_block(p1, opts, [3, 3, 128, 256], nonLin, 'stride', 1, 'pad', 1) ;
      p2 = vl_nnpool(r4, [3 3], 'stride', 2, 'pad', 1) ;
      r5 = add_block(p2, opts, [3, 3, 256, 256], nonLin, 'stride', 1, 'pad', 1) ;
      p3 = vl_nnpool(r5, [3 3], 'stride', 2, 'pad', 1) ;
      r6 = add_block(p3, opts, [3, 3, 256, 512], nonLin, 'stride', 1, 'pad', 1) ;
      p4 = vl_nnpool(r6, [3 3], 'stride', 2, 'pad', 0) ;
      r7 = add_block(p4, opts, [3, 3, 512, 512], nonLin, 'stride', 1, 'pad', 0) ;
      r8 = add_block(r7, opts, [3, 3, 512, 512], nonLin, 'stride', 1, 'pad', 0) ;

      sourceLayers = { p2, p3, p4, r7, r8 } ;  
    otherwise
      error(sprintf('%s not recognized', opts.modelOpts.archName)) ;
  end
end

% -------------------------------------
% Compute multibox prior parameters
% -------------------------------------
aspectRatios = repmat({[2, 3]}, 1, numel(sourceLayers)) ;

% select number of source layers
switch opts.modelOpts.numSources
  case 1
    sel = 1 ;
    minRatio = 10 ; maxRatio = 30 ;
  case 2
    sel = [1 2] ;
    minRatio = 10 ; maxRatio = 50 ;
  case 3
    sel = [1 2 3] ;
    minRatio = 10 ; maxRatio = 80 ;
  case 4
    sel = [1 2 3 4] ;
    minRatio = 10 ; maxRatio = 100 ;
  case 5
    sel = [1 2 3 4 5] ;
    minRatio = 10 ; maxRatio = 105 ;
  otherwise
    error(sprintf('%d source layers! Are you out of your goddamn mind??', ...
                   opts.modelOpts.numSources)) ;
end
sourceLayers = sourceLayers(sel) ; 

% These ratios are expressed as percentages
priorOpts = getPriorOpts(sourceLayers, minRatio, maxRatio, opts) ;
priorOpts.inputIm = data ;
priorOpts.pixelStep = computePixelSteps(sourceLayers{end}, sourceLayers, opts) ;

% selected by source layer
priorOpts.aspectRatios = aspectRatios(sel) ;

for unit = 1:numel(sourceLayers)
    predictors{unit} = addMultiBoxLayers(unit, priorOpts, opts) ;
end

% Fuse predictions  
concat = Layer.fromFunction(@vl_nnconcat) ;
priorBoxLayers = cellfun(@(x) {x{1}}, predictors) ;

dim = 1 ;
fusedPriors = cat(dim, priorBoxLayers{:}) ;
fusedPriors.name = 'fusedPriors' ;

dim = 3 ;
locLayers = cellfun(@(x) {x{2}}, predictors) ;
confLayers = cellfun(@(x) {x{3}}, predictors) ;
fusedLocs = cat(dim, locLayers{:}) ;
fusedLocs.name = 'fusedLocs' ;
fusedConfs = cat(dim, confLayers{:}) ;
fusedConfs.name = 'fusedConfs' ;
if opts.modelOpts.pointNet
  appLayers = cellfun(@(x) {x{4}}, predictors) ;
  fusedApps = cat(dim, appLayers{:}) ;
  fusedApps.name = 'fusedApps' ;
end

[multiloss,regloss,softmaxlog] = ssd_add_loc_conf_loss(opts,gtBoxes,gtLabels,fusedPriors,fusedConfs,fusedLocs,'');

visuals = Layer.create(@viz_ssd_coder, {data, gtLabels, gtBoxes, ...
                                 fusedPriors, fusedLocs, fusedConfs, ...
                                 'numClasses', opts.modelOpts.numClasses}, ...
                                 'numInputDer', 0) ;
visuals.name = 'visuals' ;
% --------------------------------------------------------
% Add detection subnetwork to generate mAP during training
% --------------------------------------------------------
% Note: this also means that at deployment time, we can simply
% prune the training branches

addedlosses = {multiloss, visuals};  

if opts.trackMAP
  mAP = ssd_add_map(opts,fusedConfs,fusedLocs,fusedPriors,gtLabels,gtBoxes,epoch,'');
  addedlosses = {addedlosses{:}, mAP} ;
end


if isfield(opts.modelOpts,'logAverageConf') && opts.modelOpts.logAverageConf
  avgConf = Layer.create(@mean,{abs(fusedConfs(:))},'numInputDer', 0);
  avgConf.name = 'avgConf' ;  
  addedlosses{end+1} = avgConf;
end

if opts.modelOpts.pointNet
  ptnetlosses = ssd_add_pointnet(opts,data,fusedConfs,fusedLocs,fusedPriors,fusedApps,gtLabels,gtBoxes,epoch);
  addedlosses = {addedlosses{:},ptnetlosses{:}};  
end

net = Net(addedlosses{:}) ;
net.meta.normalization.imageSize = repmat(opts.modelOpts.architecture, [1 2]) ;


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

% -----------------------------------------------------------------------
function priorOpts = getPriorOpts(sourceLayers, minRatio, maxRatio, opts) 
% -----------------------------------------------------------------------

% minimum dimension of input image
minDim = opts.modelOpts.architecture ;

% set standard options
priorOpts.kernelSize = [3 3] ;
priorOpts.permuteOrder = [3 2 1 4] ;
priorOpts.flattenAxis = 3 ;

step = floor((maxRatio - minRatio) / max((numel(sourceLayers) - 1), 1)) ;
minSizes = zeros(numel(sourceLayers), 1) ;
maxSizes = zeros(numel(sourceLayers), 1) ;
effectiveMax = minRatio + numel(sourceLayers) * step ;
ratios = [minRatio:step:maxRatio effectiveMax] ;
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

% ------------------------------------------------------------
function [predictors] = addMultiBoxLayers(unit, priorOpts, opts)
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

% if the current layer is a pooling layer, we need to search back through
% to the last conv layer to determine the current number of channels
prev = srcFeatures.find(@vl_nnconv, -1) ;
channelsIn = size(prev.inputs{3}.value, 1) ;
assert(3 <= channelsIn && channelsIn <= 1024, 'unexpected number of channels') ;

% we do not include a non linear component in the prediction blocks
nonLinearity = false ; 

% add bbox regression predictors
sz = [priorOpts.kernelSize(1:2) channelsIn locOut] ;
loc = add_block(srcFeatures, opts, sz, nonLinearity, 'stride', 1, 'pad', 1) ;
perm = Layer.create(@permute, {loc, priorOpts.permuteOrder}) ;
perm.name = sprintf('loc_perm%d', unit) ;
flatLoc = Layer.create(@vl_nnflatten, {perm, priorOpts.flattenAxis}) ;
flatLoc.name = sprintf('loc_flat%d', unit) ;

% add class confidence predictors
sz = [priorOpts.kernelSize(1:2) channelsIn confOut] ;
conf = add_block(srcFeatures, opts, sz, nonLinearity, 'stride', 1, 'pad', 1) ;
perm = Layer.create(@permute, {conf, priorOpts.permuteOrder}) ;
perm.name = sprintf('conf_perm%d', unit) ;
flatConf = Layer.create(@vl_nnflatten, {perm, priorOpts.flattenAxis}) ;
flatConf.name = sprintf('conf_flat%d', unit) ;

predictors = { priorBox, flatLoc, flatConf } ;

if opts.modelOpts.pointNet
  % appearance predictors 
  appOut = opts.modelOpts.appDim*(locOut/4);
  sz = [priorOpts.kernelSize(1:2) channelsIn appOut] ;
  app = add_block(srcFeatures, opts, sz, nonLinearity, 'stride', 1, 'pad', 1) ;
  perm = Layer.create(@permute, {app, priorOpts.permuteOrder}) ;
  perm.name = sprintf('app_perm%d', unit) ;
  flatApp = Layer.create(@vl_nnflatten, {perm, priorOpts.flattenAxis}) ;
  flatApp.name = sprintf('app_flat%d', unit) ;
  predictors{end+1} = flatApp;
end

% -------------------------------------------------------------
function net = add_block(net, opts, sz, nonLinearity, varargin)
% -------------------------------------------------------------
filters = Param('value', init_weight(opts, sz, 'single'), 'learningRate', 1) ;
biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;

net = vl_nnconv(net, filters, biases, varargin{:}, ...
                'CudnnWorkspaceLimit', opts.modelOpts.CudnnWorkspaceLimit) ;

if nonLinearity
  bn = opts.modelOpts.batchNormalization ;
  rn = opts.modelOpts.batchRenormalization ;
  assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
  if bn
    net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
  elseif rn
    net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ; 
  end
  net = vl_nnrelu(net) ;
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, sz, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

sc = sqrt(2/(sz(1)*sz(2)*sz(4))) ;  
weights = randn(sz, type)*sc ;

% -------------------------------------------------------------------------
function pixelSteps = computePixelSteps(net, sourceLayers, opts)
% -------------------------------------------------------------------------
% compute the pixel steps - so fly like a g6
trunk = Net(net) ;
imSz = opts.modelOpts.architecture ;
inputs = {'data', randn(imSz, imSz, 3, 1, 'single')} ;

if opts.batchRenormalization
  clips = [1 0] ;
  inputs = {inputs{:}, 'clips', clips}  ;
end

trunk.eval(inputs, 'forward') ;

for i = 1:numel(sourceLayers)
  layer = sourceLayers{i} ;
  feats = trunk.getValue(layer) ;
  featDim = size(feats, 1) ;
  pixelSteps(i) = imSz / featDim ;
end
