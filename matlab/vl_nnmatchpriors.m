function [matches, targets, tWeights, boxes] = vl_nnmatchpriors(p, gt, varargin) 
%VL_NNMATCHPRIORS matches prior boxes against ground truth annotations
%   Y = VL_NNMATCHPRIORS(P, GT) encodes the training
%   annotations and the network outputs into a common format 
%   from which the loss can be computed. The inputs consist of the
%   raw predictions made by the network, prior boxes and ground 
%   truth annotations.  The predictions are made *relative* to 
%   the set of prior bounding boxes.  In the following description, 
%   `pB` is used to denote the number of prior boxes used to generate
%   predictions and `gtB` is used to denote the number of ground 
%   truth annotated bounding boxes.
%
%   The input P is a C3 x 1 x 2 x N array containing the prior boxes, 
%   which are encoded as a set of bbox coordinates (xminx, ymin, 
%   xmax, ymax) and a set of four "variances" which are used to scale 
%   the resulting boxes, where C3 = 4 * B
%   
%   The input GT is a 1 x 1 x C5 x N array containing annotations 
%   
%   VL_NNMATCHPRIORS(...,'OPT',VALUE,...) takes the following options:
%
%   `overlapThreshold`:: 0.5 
%    The threshold used to determine whether a ground truth annotation is 
%    to be matched to a given prior box. The prior box then becomes a 
%    positive example during training.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.matchRanker = 'overlap' ;
  opts.overlapThreshold = 0.5 ;
  opts.ignoreXBoundaryBoxes = false ;
  [opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  assert(isempty(dzdy), 'prior matching is not performed on the back pass') ;
  cellfun(@(x) assert(all(all(x(:,3:4) - x(:,1:2) > 0)), ...
          'MULTIBOXCODER:invalidGroundTruthBoxes', ...
          'ground truth boxes must be in the (xmin, ymin, xmax, ymax) format'), ...
          gt) ;

  % --------------------------------------------
  % Decoding priors and ground truth annotations
  % --------------------------------------------
  % Priors are identical across a batch, so we only produce one set per batch
  batchSize = numel(gt) ; numPriors = size(p, 1) / 4 ;
  priors = reshape(p, numPriors, 4, 2, size(p,4)) ;
  pBoxes = permute(reshape(priors(:,:,1,:), 4, [], 1, size(p,4) ),[2 1 3 4]) ;
  pVar = permute(reshape(priors(:,:,2,:), 4, [], 1, size(p,4) ),[2 1 3 4]) ;

  pBoxes = gather(pBoxes) ;
  pCenWH = bboxCoder(pBoxes, 'MinMax', 'CenWH') ;

  matches = cellfun(@(x) matchPriors(x, pBoxes, ...
               'overlapThreshold', opts.overlapThreshold, ...
               'matchRanker', opts.matchRanker, ...
               'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes), gt) ;

  % Repeat each gt bounding box for every prior it has been matched against
  % (enables vectorization of the bbox target computation)
  gtCenWH = cellfun(@(x) {bboxCoder(x, 'MinMax', 'CenWH')}, gt) ;
  matchIdx = arrayfun(@(x) {horzcat(matches(x).idx{:})}, 1:batchSize) ;
  matchesPerGt = arrayfun(@(x) {cellfun(@numel, matches(x).idx)'}, 1:batchSize) ;
  repGtBoxesWH = arrayfun(@(x) {repeatBoxes(gtCenWH{x}, matchesPerGt{x})}, ...
                                1:batchSize) ;

  pVar = cellfun(@(x) {pVar(x, :)}, matchIdx) ;
  pCenWH = cellfun(@(x) {pCenWH(x, :)}, matchIdx) ;

  % fuse outputs
  gtBoxes = vertcat(repGtBoxesWH{:}) ;
  pBoxes = vertcat(pCenWH{:}) ;
  pVars = vertcat(pVar{:}) ;
  boxes = {gtBoxes, pBoxes, pVars} ;

  % Compute target offsets across the batch
  targets = priorCoder(gtBoxes, pBoxes, pVars, 'targets') ;

  % create a weight for each target
  tWeights = ones(size(targets)) ;

  % scale weights
  tWeights = tWeights * (1 / size(targets, 1)) ;

% ------------------------------------------
function boxes = repeatBoxes(boxes, repeats)
% ------------------------------------------
  boxes = arrayfun(@(x) {repmat(boxes(x,:), [repeats(x) 1])}, 1:numel(repeats)) ;
  boxes = vertcat(boxes{:}) ;
