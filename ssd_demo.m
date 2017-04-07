function ssd_demo(varargin)
%SSD_DEMO Minimalistic demonstration of the SSD detector

% Setup MatConvNet and SSD
curr = fileparts(mfilename('fullpath')) ;
run(fullfile(curr, '../../matlab/vl_setupnn.m')) ;
setup_mcnSSD ;

opts.modelPath = '' ;
opts.gpu = [1] ;
opts.batchSize = 1 ;

% The network is trained to prediction occurences
% of the following classes from the pascal VOC challenge
opts.classes = {'none_of_the_above', ...
    'aeroplane', ...
    'bicycle', ...
    'bird', ...
    'boat', ...
    'bottle', ...
    'bus', ...
    'car', ...
    'cat', ...
    'chair', ...
    'cow', ...
    'diningtable', ...
    'dog', ...
    'horse', ...
    'motorbike', ...
    'person', ...
    'pottedplant', ...
    'sheep', ...
    'sofa', ...
    'train', ...
    'tvmonitor'} ;

opts = vl_argparse(opts, varargin) ;

% Load or download an example SSD model:
modelName = 'ssd-mcn-pascal-vggvd-300.mat' ;
paths = {opts.modelPath, ...
         modelName, ...
         fullfile(vl_rootnn, 'data', 'models', modelName), ...
         fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
ok = min(find(cellfun(@(x) exist(x, 'file'), paths))) ;

if isempty(ok)
  fprintf('Downloading the SSD model ... this may take a while\n') ;
  opts.modelPath = fullfile(vl_rootnn, 'data/models', modelName) ;
  mkdir(fileparts(opts.modelPath)) ;
  url = sprintf('http://www.robots.ox.ac.uk/~albanie/models/ssd/%s', modelName) ;
  urlwrite(url, opts.modelPath) ;
else
  opts.modelPath = paths{ok} ;
end

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;

% Load test image
im = single(imread('test.jpg')) ;
im = imresize(im, [300 300]) ;


% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  net.move('gpu') ;
  im = gpuArray(im) ;
end

% tell the network to store the outputs
% of the prediction layer and do a forward pass
net.mode = 'test';
net.vars(end).precious = true ;
net.eval({'data', im}) ;

% gather the predictions from the network,
% and sort by confidence
preds = net.vars(end).value ;

% check the last 
preds = preds(:,:,:,end) ;

[~, sortedIdx ] = sort(preds(:, 2), 'descend') ;
preds = preds(sortedIdx, :) ;

% extract the most confident prediction
box = preds(1,3:end) ;
confidence = preds(1,2) ;
label = opts.classes{preds(1,1)} ;

% return image to cpu for visualisation
if numel(opts.gpu) > 0
  im = gather(im) ;
end

% diplay prediction as a sanity check
figure ;
im = im / 255 ;
x = box(1) * size(im, 2) ;
y = box(2) * size(im, 1) ;
width = box(3) * size(im, 2) - x ;
height = box(4) * size(im, 1) - y ;
rectangle = [x y width height];
im = insertShape(im, 'Rectangle', rectangle, ...
                  'LineWidth', 3, ...
                  'Color', 'red');
imagesc(im) ;
title(sprintf('top SSD prediction: %s \n confidence: %f', label, confidence)) ;

% free up the GPU allocation
if numel(opts.gpu) > 0
  net.move('cpu') ;
end
