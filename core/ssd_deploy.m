function net = ssd_deploy(srcPath, destPath, numClasses)
% SSD_DEPLOY deploys an SSD model for evaluation
%   NET = SSD_DEPLOY(SRCPATH, DESTPATH, NUMCLASSES) configures
%   an SSD model to perform evaluation.  THis process involves
%   removing the loss layers used during training and adding 
%   a combination of a transpose softmax with a detection
%   layer to compute network predictions
%
% Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

tmp = load(srcPath) ;
net = dagnn.DagNN.loadobj(tmp.net) ;

outDir = fileparts(destPath) ;
if ~exist(outDir, 'dir')
    mkdir(outDir) ;
end

% clear prior caches
priorBoxLayerIdx = find(cellfun(@(x) isa(x, 'dagnn.PriorBox'), ...
                                                {net.layers.block})) ;
for i = 1:numel(priorBoxLayerIdx)
    net.layers(priorBoxLayerIdx(i)).block.priorCache = {} ;
end


% remove training layers, replace with detection output
net.removeLayer('loc_loss') ;
net.removeLayer('class_loss') ;
net.removeLayer('mbox_coder') ;
net.removeLayer('mbox_loss') ;

% Add layers used for detection 
layerName = 'mbox_conf_reshape' ;
layer = dagnn.Reshape('shape', {numClasses, [], 1}) ;
inputs = {'mbox_conf'} ; 
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = 'mbox_conf_softmaxt' ;
layer = dagnn.SoftMaxTranspose() ;
inputs = {'mbox_conf_reshape'} ; 
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = 'mbox_conf_flatten' ;
layer = dagnn.Flatten('axis', 3 ) ;
inputs = {'mbox_conf_softmaxt'} ; 
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

layerName = 'detection_out' ;
layer = dagnn.MultiboxDetector('backgroundLabel', 1, 'numClasses', numClasses) ;
inputs = {'mbox_loc', 'mbox_conf_flatten', 'mbox_priorbox'} ; 
outputs = layerName ;
params = {} ;
net.addLayer(layerName, layer, inputs, outputs, params) ;

net.rebuild() ;
net.meta.backgroundClass = 1 ;


net = net.saveobj() ;
save(destPath, '-struct', 'net') ;
