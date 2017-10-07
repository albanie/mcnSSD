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

  outDir = fileparts(destPath) ;
  if ~exist(outDir, 'dir'), mkdir(outDir) ; end
  tmp = load(srcPath) ; 
  if ~isfield(tmp.net, 'foward') % support dagnn checkpoints
    dag = dagnn.DagNN.loadobj(tmp.net) ;
    dag.removeLayer('loc_loss') ;
    dag.removeLayer('class_loss') ;
    dag.removeLayer('mbox_coder') ;
    dag.removeLayer('mbox_loss') ;
    stored = Layer.fromDagNN(dag, @extras_autonn_custom_fn) ; 
  else
    stored = Layer.fromCompiledNet(tmp.net) ;
  end

  priors = findByName(stored, 'mbox_priorbox') ;
  confs = findByName(stored, 'mbox_conf') ;
  locs = findByName(stored, 'mbox_loc') ;

  shape = {numClasses, 1, []} ;
  flattenAxis = 3 ; softDim = 1 ; nmsThresh = 0.45 ;
  res = Layer.create(@vl_nnreshape, {confs, shape}) ;
  softT = Layer.create(@vl_nnsoftmaxt, {res, 'dim', softDim}) ;
  flat = Layer.create(@vl_nnflatten, {softT, flattenAxis}) ;
  det = Layer.create(@vl_nnmultiboxdetector, {locs, flat, priors, ...
                                     'numClasses', numClasses, ...
                                     'nmsThresh', nmsThresh}) ;
  det.name = 'detection_out' ; net = Net(det) ; net.meta = tmp.net.meta ;
  % add standard imagenet average if not present
  if ~isfield(net.meta, 'normalization'), net.meta.normalization = struct() ; end
  if ~isfield(net.meta.normalization, 'averageImage') || ...
    isempty(net.meta.normalization.averageImage)
    rgb = [122.771, 115.9465, 102.9801] ; 
    net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;
  end
  net.meta.backgroundClass = 1 ; 
  net = net.saveobj() ;
  save(destPath, '-struct', 'net') ;

% ---------------------------------------
function layer = findByName(stored, name)
% ---------------------------------------

  for ii = 1:numel(stored)
    if ~isempty(stored{ii}.find(name))
      layer = stored{ii}.find(name, 1) ; return ;
    end
  end
  error('layer %s not found in network\n', name) ;
