% reload paths etc.
% NOTE!
% The mex lock has been removed from caffe_.cpp! 
%munlock('caffe_') ;
clear all ; %#ok
caffePath = '/users/albanie/coding/libs/caffes/ssd-caffe/matlab' ;
%munlock('caffe_') ;
%clear caffe_.mexa64 ;
clear caffe_ ;
rehash path ;

% hack to avoid cuda errors
a = gpuArray(1); clear a ; %#ok

% refresh/set up ssd-caffe
addpath(caffePath) ;

caffeRoot = '/users/albanie/coding/libs/caffes/ssd-caffe' ;

% set paths and load caffe ssd-model
modelDir = '/users/albanie/data/models/caffe/mobilenet-ssd' ;
dataDir = '/users/albanie/data/models/caffe/mobilenet-ssd' ;
model = fullfile(modelDir, 'MobileNetSSD_deploy.proto') ;
weights = fullfile(dataDir, 'MobileNetSSD_deploy.caffemodel') ;
caffe.set_mode_cpu() ;

% to use the relative paths defined in the prototxt, we change into the 
% caffe root to load the network
cd(caffeRoot) ;

% load model
caffeNet = caffe.Net(model, weights, 'test') ;

% load mcn model for comparison

% set path to trunk model (VGG-VD-16)
opts.trunkModelPath = fullfile(vl_rootnn, 'data', 'models-import', ...
                                'ssd-mcn-mobilenet.mat') ;
opts.imageSize = [300 300] ;
net = load(opts.trunkModelPath) ; net = dagnn.DagNN.loadobj(net) ;
