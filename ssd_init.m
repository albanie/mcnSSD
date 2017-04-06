function net = ssd_init(opts)
% SSD_INIT Initialize a Single Shot Multibox Detector Network
%   NET = SSD_INIT randomly initializes an SSD network architecture

trunkModelPath = fullfile(vl_rootnn, ...
                          'data/models-import/vgg-vd-16-reduced.mat') ;

% for reproducibility, fix the seed
rng(0, 'twister') ;

% load pre-trained base network 
net = vl_simplenn_tidy(load(trunkModelPath)) ;

% -------------------------------------------------------------
% Set meta properties of net
% -------------------------------------------------------------

switch opts.modelOpts.architecture
    case 300
        imSize = [300 300] ;
        addExtraConv = false ;
    case 500
        imSize = [500 500] ;
        addExtraConv = true ;
    case 512
        imSize = [512 512] ;
        addExtraConv = true ;
    otherwise
        error('architecture %d is not recognised', ...
                             opts.modelOpts.architecture) ;
end

net.meta.normalization.imageSize = imSize ;

% -------------------------------------------------------------
% Network truncation
% -------------------------------------------------------------
net = dagnn.DagNN.fromSimpleNN(net) ;

% modify trunk biases learnning rate and weight decay to match caffe 
params = {'conv1_1b', ...
          'conv1_2b', ...
          'conv2_1b', ...
          'conv2_2b', ... 
          'conv3_1b', ...
          'conv3_2b', ...
          'conv3_3b', ...
          'conv4_1b', ...
          'conv4_2b', ...
          'conv4_3b' ...
          'conv5_1b', ...
          'conv5_2b', ...
          'conv5_3b' ...
          'fc6b', ...
          'fc7b' ...
          } ;
for i = 1:length(params)
    net = matchCaffeBiases(net, params{i}) ;
end

% update fc6 to match SSD
net.layers(net.getLayerIndex('fc6')).block.dilate = [6 6] ;
net.layers(net.getLayerIndex('fc6')).block.pad = [6 6] ;

% Truncate the layers following relu7
net.removeLayer('fc8') ;
net.removeLayer('prob') ;

% rename input variable to 'data'
net.renameVar('x0', 'data') ;

% Alter the pooling to match SSD
net.layers(net.getLayerIndex('pool5')).block.stride = [1 1] ;
net.layers(net.getLayerIndex('pool5')).block.poolSize = [3 3] ;
net.layers(net.getLayerIndex('pool5')).block.pad = [1 1 1 1] ;

% --------------------------------------------------------------
% Architectural modifications - add new conv layer stacks
% --------------------------------------------------------------

prefix = 'conv6' ;
inLayer = 'relu7' ;
channelsIn = 1024 ;
bottleneck = 256 ;
channelsOut = 512 ;
kernelSizes = [1 3] ;
paddings = [0 0 0 0 ; 1 1 1 1] ;
strides = [ 1 1 ; 2 2 ] ;
net = addConvStack(net, prefix, inLayer, channelsIn, bottleneck, ...
                   channelsOut, kernelSizes, paddings, strides) ; 

prefix = 'conv7' ;
inLayer = 'conv6_2_relu' ;
channelsIn = 512 ;
bottleneck = 128 ;
channelsOut = 256 ;
kernelSizes = [1 3] ;
paddings = [0 0 0 0 ; 1 1 1 1] ;
strides = [ 1 1 ; 2 2 ] ;
net = addConvStack(net, prefix, inLayer, channelsIn, bottleneck, ...
                   channelsOut, kernelSizes, paddings, strides) ; 

prefix = 'conv8' ;
inLayer = 'conv7_2_relu' ;
finalLayer = 'conv8_2_relu' ;
channelsIn = 256 ;
bottleneck = 128 ;
channelsOut = 256 ;
kernelSizes = [1 3] ;
paddings = [0 0 0 0 ; 0 0 0 0] ;
strides = [ 1 1 ; 1 1 ] ;
net = addConvStack(net, prefix, inLayer, channelsIn, bottleneck, ...
                   channelsOut, kernelSizes, paddings, strides) ; 

prefix = 'conv9' ;
inLayer = 'conv8_2_relu' ;
finalLayer = 'conv9_2_relu' ;
channelsIn = 256 ;
bottleneck = 128 ;
channelsOut = 256 ;
kernelSizes = [1 3] ;
paddings = [0 0 0 0 ; 0 0 0 0] ;
strides = [ 1 1 ; 1 1 ] ;
net = addConvStack(net, prefix, inLayer, channelsIn, bottleneck, ...
                   channelsOut, kernelSizes, paddings, strides) ; 

if addExtraConv
    prefix = 'conv10' ;
    inLayer = 'conv9_2_relu' ;
    finalLayer = 'conv10_2_relu' ;
    channelsIn = 256 ;
    bottleneck = 128 ;
    channelsOut = 256 ;
    kernelSizes = [1 4] ;
    paddings = [0 0 0 0 ; 1 1 1 1] ;
    strides = [ 1 1 ; 1 1 ] ;
    net = addConvStack(net, prefix, inLayer, channelsIn, bottleneck, ...
                   channelsOut, kernelSizes, paddings, strides) ; 
end

% Add normalization layer 
layerName = 'conv4_3_norm' ;
initialScale = single(20) ;
layer = dagnn.Normalize('channelShared', false, ...
                        'acrossSpatial', false) ;
inputs = net.layers(net.getLayerIndex('relu4_3')).outputs ;
outputs = layerName ;
params = {'conv4_3_norm_scale'} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;
net.params(net.getParamIndex(params)).value = ones(1,1,512) * initialScale ;

% -------------------------------------
% Compute multibox prior parameters
% -------------------------------------

% minimum dimension of input image
minDim = imSize(1) ;

switch opts.modelOpts.architecture
    case 300

        % The feature maps which from which the prior box layers are 
        % constructed have the following sizes:
        sourceLayers = {'conv4_3_norm', ... % 38 x 38
                        'fc7',  ...         % 19 x 19
                        'conv6_2', ...      % 10 x 10
                        'conv7_2', ...      % 5 x 5 
                        'conv8_2',  ...     % 3 x 3
                        'conv9_2'} ;        % 1 x 1

        % These ratios are expressed as percentages
        minRatio = 20 ; 
        maxRatio = 90 ;

        % Following the paper, the scale for conv4_3 is handled separately
        % when training on VOC0712, so we can compute the step as follows
        step = floor((maxRatio - minRatio) / (numel(sourceLayers(2:end)) - 1)) ;

        minSizes = zeros(numel(sourceLayers), 1) ;
        maxSizes = zeros(numel(sourceLayers), 1) ;

        effectiveMax = minRatio + numel(sourceLayers(2:end)) * step ;
        ratios = [10 minRatio:step:maxRatio effectiveMax] ;

        for i = 1:numel(ratios) - 1 
            minSizes(i) = minDim * ratios(i) / 100 ;
            maxSizes(i) = minDim * ratios(i + 1) / 100 ;
        end

        % note: certain layers use fewer aspect ratios (just 1, 1/2, 2/1)
        aspectRatios = { [2], ... 
                         [2, 3], ...
                         [2, 3], ...
                         [2, 3], ...
                         [2], ...
                         [2] } ;

        steps = [8, 16, 32, 64, 100, 300] ;

    case 512

        % The feature maps which from which the prior box layers are 
        % constructed have the following sizes:
        sourceLayers = {'conv4_3_norm', ... % 64 x 64
                        'fc7',  ...         % 32 x 32
                        'conv6_2', ...      % 16 x 16
                        'conv7_2', ...      % 8 x 8 
                        'conv8_2',  ...     % 4 x 4
                        'conv9_2', ...      % 2 x 2
                        'conv10_2'} ;       % 1 x 1

        % These ratios are expressed as percentages
        minRatio = 15 ; 
        maxRatio = 90 ;

        step = floor((maxRatio - minRatio) / (numel(sourceLayers(2:end)) - 1)) ;

        minSizes = zeros(numel(sourceLayers), 1) ;
        maxSizes = zeros(numel(sourceLayers), 1) ;

        effectiveMax = minRatio + numel(sourceLayers(2:end)) * step ;
        ratios = [7 minRatio:step:maxRatio effectiveMax] ;

        for i = 1:numel(ratios) - 1 
            minSizes(i) = minDim * ratios(i) / 100 ;
            maxSizes(i) = minDim * ratios(i + 1) / 100 ;
        end

        aspectRatios = { [2], ... 
                         [2, 3], ...
                         [2, 3], ...
                         [2, 3], ...
                         [2, 3], ...
                         [2], ...
                         [2] } ;

        steps = [8, 16, 32, 64, 128, 256, 512] ;
end

flip = true ;
clip = opts.modelOpts.clipPriors ;
variances = [0.1 0.1 0.2 0.2]' ;

% an offset is used to centre each prior box between  
% activations in the feature map (see Sec 2.2. of the 
% SSD paper)
offset = 0.5 ;

% Since the computed prior boxes are identical for a fixed size
% input, we can cahce them after the first forward pass
usePriorCaching = true ;

% -------------------------------------
% Construct multibox prior units
% -------------------------------------
unit = 1 ;
prefix = 'conv4_3_norm' ;
inLayerName = 'conv4_3_norm' ;
channelsIn = 512 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

unit = 2 ;
prefix = 'fc7' ;
inLayerName = 'relu7' ;
channelsIn = 1024 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

unit = 3 ;
prefix = 'conv6_2' ;
inLayerName = 'conv6_2_relu' ;
channelsIn = 512 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

unit = 4 ;
prefix = 'conv7_2' ;
inLayerName = 'conv7_2_relu' ;
channelsIn = 256 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

unit = 5 ;
prefix = 'conv8_2' ;
inLayerName = 'conv8_2_relu' ;
channelsIn = 256 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

unit = 6 ;
prefix = 'conv9_2' ;
inLayerName = 'conv9_2_relu' ;
channelsIn = 256 ;
[confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                        maxSizes(unit), aspectRatios{unit}, flip, ...
                        clip, steps(unit), offset, variances, ...
                        channelsIn, confOut, locOut, usePriorCaching) ;

if addExtraConv
    unit = 7 ;
    prefix = 'conv10_2' ;
    inLayerName = 'conv10_2_relu' ;
    channelsIn = 256 ;
    [confOut, locOut] = getOutSize(maxSizes(unit), aspectRatios{unit}, opts) ;
    net = addMultiBoxLayers(net, prefix, inLayerName, minSizes(unit), ...
                            maxSizes(unit), aspectRatios{unit}, flip, ...
                            clip, steps(unit), offset, variances, ...
                            channelsIn, confOut, locOut, usePriorCaching) ;
end

% -------------------------------------
% Fuse predictions  
% -------------------------------------

layerName = 'mbox_loc' ;
fuseType = 'loc_flat' ;
axis = 3 ;
net = addFusionLayer(net, layerName, fuseType, sourceLayers, axis) ;

layerName = 'mbox_conf' ;
fuseType = 'conf_flat' ;
axis = 3 ;
net = addFusionLayer(net, layerName, fuseType, sourceLayers, axis) ;

layerName = 'mbox_priorbox' ;
fuseType = 'priorbox' ;
axis = 1 ;
net = addFusionLayer(net, layerName, fuseType, sourceLayers, axis) ;

% ----------------------------------------------------------------
% Decoder layer
% ----------------------------------------------------------------

% Decoder layer -> converts predictions into happiness and rainbows
layerName = 'mbox_coder' ;
layer = dagnn.MultiboxCoder('backgroundLabel', 1,  ...
                            'locWeight', 1,  ...
                            'normalization', 'VALID', ...
                            'negOverlap', 0.5, ...
                            'overlapThreshold', 0.5, ...
                            'negPosRatio', 3, ...
                            'numClasses', opts.modelOpts.numClasses, ...
                            'hardNegativeMining', true) ;
inputLayers = {'mbox_loc', 'mbox_conf', 'mbox_priorbox'} ;
inputs = cellfun(@(x) net.layers(net.getLayerIndex(x)).outputs{1}, ...
                    inputLayers, 'UniformOutput', false) ;
inputs = horzcat(inputs, 'targets', 'labels') ;
outputs = {'loc_preds', 'loc_labels', 'conf_preds', ...
           'conf_labels', 'loc_weights', 'conf_weights'} ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

% ----------------------------------------------------------------
% Loss layers
% ----------------------------------------------------------------
% Two losses are used

layerName = 'class_loss' ;
layer = dagnn.UnnormalizedLoss() ;
inputs = { 'conf_preds', 'conf_labels', 'conf_weights'} ;
outputs = {'class_loss'} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = 'loc_loss' ;
layer = dagnn.HuberLoss() ;
inputs = { 'loc_preds', 'loc_labels', 'loc_weights'} ;
outputs = {'loc_loss'} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = 'mbox_loss' ;
layer = dagnn.MultiboxLoss() ;
inputs = { 'class_loss', 'loc_loss'} ;
outputs = {'mbox_loss'} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

% ----------------------------------------------
function net = matchCaffeBiases(net, param)
% ----------------------------------------------
% set the learning rate and weight decay of the 
% convolution biases to match caffe

net.params(net.getParamIndex(param)).learningRate = 2 ;
net.params(net.getParamIndex(param)).weightDecay = 0 ;

% -------------------------------------------------------
function [confOut, locOut] = getOutSize(maxSize, aspectRatios, opts)
% -------------------------------------------------------
% THe filters must produce a prediction for each of the prior
% boxes associated with a source feature layer.  The number of 
% prior boxes per feature is explained in detail in the SSD paper 
% (essentially it is computed according to the number of specified 
% aspect ratios)

numBBoxOffsets = 4 ;
priorsPerFeature = 1 + boolean(maxSize) + numel(aspectRatios) * 2 ;
confOut = opts.modelOpts.numClasses *  priorsPerFeature ;
locOut = numBBoxOffsets *  priorsPerFeature ;

% -----------------------------------------------------------------------------
function net = addConvStack(net, prefix, prevLayer, channelsIn, bottleneck, ...
                            channelsOut, kernelSizes, paddings, strides) 
% -----------------------------------------------------------------------------

layerName = sprintf('%s_1', prefix) ;
ks = kernelSizes(1) ; 
sz = [ks ks channelsIn bottleneck] ;
layer = dagnn.Conv('size', sz, ...
                    'pad', paddings(1,:), ...
                    'stride', strides(1,:), ...
                    'hasBias', true) ;
inputs = net.layers(net.getLayerIndex(prevLayer)).outputs ;
outputs = layerName ;
params = {sprintf('%s_1f', prefix), sprintf('%s_1b', prefix)} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;
net = init_weights(net, layerName, ks, channelsIn, bottleneck) ;


prevLayer = layerName ;
layerName = sprintf('%s_1_relu', prefix) ;
layer = dagnn.ReLU() ;
inputs = net.layers(net.getLayerIndex(prevLayer)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

prevLayer = layerName ;
layerName = sprintf('%s_2', prefix) ;
ks = kernelSizes(2) ; 
sz = [ks ks bottleneck channelsOut] ;
layer = dagnn.Conv('size', sz, ...
                    'pad', paddings(2,:), ...
                    'stride', strides(2,:), ...
                    'hasBias', true) ;
inputs = net.layers(net.getLayerIndex(prevLayer)).outputs ;
outputs = layerName ;
params = {sprintf('%s_12', prefix), sprintf('%s_2b', prefix)} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;
net = init_weights(net, layerName, ks, bottleneck, channelsOut) ;

prevLayer = layerName ;
layerName = sprintf('%s_2_relu', prefix) ;
layer = dagnn.ReLU() ;
inputs = net.layers(net.getLayerIndex(prevLayer)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

% -------------------------------------------------------------------
function net = addMultiBoxLayers(net, prefix, prevLayerName, minSize, ... 
                                 maxSize, aspectRatios, flip, clip, ...
                                 step, offset, variances, channelsIn, ...
                                 confOut, locOut, usePriorCaching) 
% ------------------------------------------------------------------
%ADDMULTIBOXLAYERS adds a set of multibox layers to a dagnn
%   ADDMULTIBOXLAYERS adds the collection of network layers
%   required to construct a MultiBox Prior "unit". A unit 
%   consistss of seven layers:
%   
%   A PriorBox reference layer defining the default
%   reference boxes, followed by 
%       Convolution -> Permute -> Flatten 
%           (for prior box locations)
%       Convolution -> Permute -> Flatten 
%           (for prior box class scores)
%       
%   ARGUMENTS:
%       `net`: dagnn.DagNN net object under construction
%       `prefix`: (string) name used to prefix each layer name in unit
%       `prevLayerName`: (string) name of input feature layer 
%       `minSize`: (float) minimum size of prior boxes 
%       `maxSize`: (float) maximum size of prior boxes 
%       `aspectRatios`: (float) array of aspect ratios used for prior boxes 
%       `flip`: (boolean) add flipped versions of each aspect ratio
%       `clip`: (boolean) clip prior boxes to lie inside feature map
%       `step`: (int) number of pixel steps taken in image to match a pixel 
%               step in the feature map
%       `offset`: (float) offset applied to centre each feature location
%       `variances`: (4x1 array) variances used to scale prior boxes
%       `channelsIn`: (int) number of channels in the input layer
%       `confOut`: (int) number of filters used to predict class confidences
%       `locOut`: (int) number of filters used to predict box location updates
%       `usePriorCaching`: (boolean) cache the prior boxes and re-use after 
%                          the first pass

% store the layer on which the multibox layers will be based
rootLayerName = prevLayerName ;

layerName = sprintf('%s_mbox_priorbox', prefix) ;
layer = dagnn.PriorBox('minSize', minSize, ...
                       'maxSize', maxSize, ...
                       'aspectRatios', aspectRatios, ...
                       'flip', flip, ...
                       'clip', clip, ...
                       'pixelStep', step, ...
                       'offset', offset, ...
                       'usePriorCaching', true, ...
                       'variance', variances) ;
inputs = { net.layers(net.getLayerIndex(rootLayerName)).outputs{1} 'data' };
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = sprintf('%s_mbox_loc', prefix) ;
ks = 3 ;
sz = [ks ks channelsIn locOut] ;
layer = dagnn.Conv('size', sz, 'pad', 1, 'stride', 1, 'hasBias', true) ;
inputs = net.layers(net.getLayerIndex(rootLayerName)).outputs ;
outputs = layerName ;
params = {sprintf('%s_mbox_locf', prefix), sprintf('%s_mbox_locb', prefix)} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;
net = init_weights(net, layerName, ks, channelsIn, locOut) ;

prevLayerName = layerName ;
layerName = sprintf('%s_mbox_loc_perm', prefix) ;
layer = dagnn.Permute('order', [3 2 1 4]);
inputs = net.layers(net.getLayerIndex(prevLayerName)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

prevLayerName = layerName ;
layerName = sprintf('%s_mbox_loc_flat', prefix) ;
layer = dagnn.Flatten('axis', 3) ;
inputs = net.layers(net.getLayerIndex(prevLayerName)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = sprintf('%s_mbox_conf', prefix) ;
ks = 3 ;
sz = [ks ks channelsIn confOut] ;
layer = dagnn.Conv('size', sz, 'pad', 1, 'stride', 1, 'hasBias', true) ;
inputs = net.layers(net.getLayerIndex(rootLayerName)).outputs ;
outputs = layerName ;
params = {sprintf('%s_mbox_conff', prefix), sprintf('%s_mbox_confb', prefix)} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;
net = init_weights(net, layerName, ks, channelsIn, confOut) ;

prevLayerName = layerName ;
layerName = sprintf('%s_mbox_conf_perm', prefix) ;
layer = dagnn.Permute('order', [3 2 1 4]);
inputs = net.layers(net.getLayerIndex(prevLayerName)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

prevLayerName = layerName ;
layerName = sprintf('%s_mbox_conf_flat', prefix) ;
layer = dagnn.Flatten('axis', 3) ;
inputs = net.layers(net.getLayerIndex(prevLayerName)).outputs ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

% --------------------------------------------------------------------------
function net = addFusionLayer(net, layerName, fuseType, sourcePrefixes, axis)
% --------------------------------------------------------------------------
%ADDFUSIONLAYER adds a layer to concatenate source layers
%   ADDFUSIONLAYER adds a concatenation layer fusing inputs of a 
%   particular type
%   
%   ARGUMENTS:
%       `layerName` name of created layer 
%       `fuseType` types to be fused (can be 'loc', 'conf' or 
%        'priorbox') 
%       `sourcPrefixes` cell array of prefix strings of layers to 
%        be fused
%       `axis` dimension of concatenation 

layer = dagnn.Concat('dim', axis) ;
inputLayers = cellfun(@(x) sprintf('%s_mbox_%s', x, fuseType), ...
                        sourcePrefixes, 'UniformOutput', false) ;
inputs = cellfun(@(x) net.layers(net.getLayerIndex(x)).outputs{1}, ...
                    inputLayers, 'UniformOutput', false) ;
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

% ------------------------------------------------------
function net = init_weights(net, layerName, ks, in, out)
% ------------------------------------------------------
% xavier initialize weights and set initial learning rates

paramNames = net.layers(net.getLayerIndex(layerName)).params ;
filterName = paramNames{1} ;
biasName = paramNames{2} ;

% mimic caffe version
sc = sqrt(1/(ks*ks*in)) ;
filters = randn(ks, ks, in, out, 'single') * sc ;
net.params(net.getParamIndex(filterName)).value = filters ;
net.params(net.getParamIndex(filterName)).learningRate = 1 ;

biases = zeros(out, 1, 'single') ;
net.params(net.getParamIndex(biasName)).value = biases ;
net.params(net.getParamIndex(biasName)).learningRate = 2 ;
net.params(net.getParamIndex(biasName)).weightDecay = 0 ;

