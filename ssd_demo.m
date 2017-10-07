function ssd_demo(varargin)
%SSD_DEMO Minimalistic demonstration of the SSD detector
%   SSD_DEMO an object detection demo with a Single Shot Detector
%
%   SSD_DEMO(..., 'option', value, ...) accepts the following
%   options:
%
%   `modelPath`:: ''
%    Path to a valid R-FCN matconvnet model. If none is provided, a model
%    will be downloaded.
%
%   `gpu`:: []
%    Device on which to run network 
%
%   `wrapper`:: 'autonn'
%    The matconvnet wrapper to be used (both dagnn and autonn are supported) 
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpu = [] ;
  opts.modelPath = '' ;
  opts.wrapper = 'autonn' ;
  opts = vl_argparse(opts, varargin) ;

  % The network is trained to prediction occurences
  % of the following classes from the pascal VOC challenge
  classes = {'background', 'aeroplane', 'bicycle', 'bird', ...
     'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
     'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
     'sofa', 'train', 'tvmonitor'} ;

  % Load or download an example SSD model:
  modelName = 'ssd-mcn-pascal-vggvd-300.mat' ;
  paths = {opts.modelPath, ...
           modelName, ...
           fullfile(vl_rootnn, 'data/models', modelName), ...
           fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
  ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

  if isempty(ok)
    fprintf('Downloading the SSD model ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models', modelName) ;
    mkdir(fileparts(opts.modelPath)) ;
    base = 'http://www.robots.ox.ac.uk/~albanie/models/ssd/%s' ;
    url = sprintf(base, modelName) ; urlwrite(url, opts.modelPath) ;
  else
    opts.modelPath = paths{ok} ;
  end

  % Load the network with the chosen wrapper
  net = loadModel(opts) ;

  % Load test image
  im = single(imread('misc/test.jpg')) ; numKeep = 2 ; 
  im = imresize(im, net.meta.normalization.imageSize(1:2)) ;

  % Evaluate network either on CPU or GPU.
  if numel(opts.gpu) > 0
    gpuDevice(opts.gpu) ; net.move('gpu') ; im = gpuArray(im) ;
  end

  % set inputs and run network
  switch opts.wrapper
    case 'dagnn' 
      net.eval({'data', im}) ;
      preds = net.vars(end).value ;
    case 'autonn'
      net.eval({'data', im}, 'test') ;
      preds = net.getValue('detection_out') ;
  end

  [~, sortedIdx ] = sort(preds(:, 2), 'descend') ;
  preds = preds(sortedIdx, :) ;

  % Extract the most confident predictions
  box = double(preds(1:numKeep,3:end)) ;
  confidence = preds(1:numKeep,2) ;
  label = classes(preds(1:numKeep,1)) ;

  % Return image to cpu for visualisation
  if numel(opts.gpu) > 0, im = gather(im) ; end

  % Diplay prediction as a sanity check
  figure(1) ; im = im / 255 ; CM = spring(numKeep); 
  x = box(:,1) * size(im, 2) ; y = box(:,2) * size(im, 1) ;
  width = box(:,3) * size(im, 2) - x ; height = box(:,4) * size(im, 1) - y ;
  rectangle = [x y width height];
  im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 4, ...
                     'Color', CM(1:numKeep,:)) ;
  imagesc(im) ;
  for ii = 1:numKeep
    str = sprintf('%s: %.2f', label{ii}, confidence(ii)) ;
    text(x(ii), y(ii)-10, str, 'FontSize', 14, ...
        'BackgroundColor', CM(ii,:)) ;
  end
  title(sprintf('SSD predictions (top %d are displayed)', numKeep), ...
                   'FontSize', 15) ;
  axis off ;
  
  % Free up the GPU allocation
  if numel(opts.gpu) > 0, net.move('cpu') ; end

% ----------------------------
function net = loadModel(opts)
% ----------------------------
  net = load(opts.modelPath) ; 
  if ~isfield(net, 'forward') % dagnn loader
    net = dagnn.DagNN.loadobj(net) ;
    switch opts.wrapper
      case 'dagnn' 
        net.mode = 'test' ; 
      case 'autonn'
        out = Layer.fromDagNN(net, @extras_autonn_custom_fn) ; 
        net = Net(out{:}) ;
    end
  else % load directly using autonn
    net = Net(net) ;
  end
