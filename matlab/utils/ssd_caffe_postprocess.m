function ssd_caffe_postprocess(modelPath, mcnRoot)
%SSD_CAFFE_POSTPROCESS configure caffe models for matconvnet
% SSD_CAFFE_POSTPROCESS(MODELPATH, MCNROOT) performs the final 
% stages of configuration for importing a caffe trained SSD 
% model into matconvnet.  
%
%   MODELPATH is a string pointing to the location of the 
%   imported caffe model
%
%   MCNROOT is a string pointing to the root directory of the 
%   matconvnet install
%
% Copyright (C) 2017 Samuel Albanie 

% set up paths
addpath(fullfile(mcnRoot,'matlab')) ;
addpath(genpath(fullfile(mcnRoot, 'examples/ssd'))) ;
vl_setupnn ;

% modify reshaping layer to work with matconvnet
net = load(modelPath) ;
reshapeIdx = find(strcmp({net.layers.name}, 'mbox_conf_reshape')) ;
numClasses = net.layers(reshapeIdx).block.shape(1) ;
net.layers(reshapeIdx).block.shape = {numClasses 1 [] } ;

% switch from softmax to softmax transpose
softmaxIdx = find(strcmp({net.layers.name}, 'mbox_conf_softmax')) ;
output = {'mbox_conf_softmaxt'} ;
net.layers(softmaxIdx).name = 'mbox_conf_softmaxt' ;
net.layers(softmaxIdx).type = 'dagnn.SoftMaxTranspose' ;
net.layers(softmaxIdx).outputs = output ;

% rename input to flatten layer
flattenIdx = find(strcmp({net.layers.name}, 'mbox_conf_flatten')) ;
net.layers(flattenIdx).inputs = output ;

% save modifications
save(modelPath, '-struct', 'net') ;
fprintf('saved configured SSD detector to %s \n', modelPath) ;
