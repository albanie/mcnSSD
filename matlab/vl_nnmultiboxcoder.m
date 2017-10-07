function [targetPreds, classPreds, tWeights] = vl_nnmultiboxcoder(x, v, m, n, gt, l, varargin) 
%VL_NNMULTIBOXCODER encodes/decodes labels/predictions into locations
%   Y = VL_NNMULTIBOXCODER(X, V, P, GT, L) encodes the training
%   annotations and the network outputs into a common format 
%   from which the loss can be computed. The inputs consist of the
%   raw predictions made by the network, prior boxes and ground 
%   truth annotations.  The predictions are made *relative* to 
%   the set of prior bounding boxes.  In the following description, 
%   `pB` is used to denote the number of prior boxes used to generate
%   predictions and `gtB` is used to denote the number of ground 
%   truth annotated bounding boxes.
%
%   The input X is a 1 x 1 x C1 x N array containing the location 
%   predictions of the network (these predictions take the form of 
%   encoded spatial updates to the prior boxes), where N is the
%   batch size and C1 = 4 * B. 
%   
%   The input V is a 1 x 1 x C2 x N  array containing the per-class 
%   confidence predictions of the network where C2 = numClasses * 21.
%   
%   The input GT is a 1 x 1 x C5 x N array containing the 
%   The input L is a 1 x 1 x C4 x N array containing the 
%
%   The output y is a 1 x 6 cell array whose values are the following:
%   outputs{1} is a vector of encoded location predictions
%   outputs{3} is a vector of instance weights used to compute the bbox loss
%   outputs{4} is a vector of class predictions
%   outputs{5} is a vector of ground truth class labels
%   outputs{6} is a vector of matched positive prior box indices
%   outputs{7} is a vector of matched negative prior box indices
%
%   [DX, DV] = VL_NNMULTIBOXENCODER(X, V, P, GT, L, DY) computes the 
%   derivatives ot the multibox encoder. DX and DV have the same
%   dimensions as X and V. 
%
%   VL_NNMULTIBOXENCODER(...,'OPT',VALUE,...) takes the following options:
%
%   `numClasses`:: 21
%    The number of classes predicted by the network (used to decode the
%    inputs X and V). 
%
%   `matchingPosIndices`:: [] 
%    The indices of the positively matched prior boxes (only used in 
%    backward pass.
%
%   `matchingNegIndices`:: [] 
%    The indices of the negatively matched prior boxes (only used in 
%    backward pass.

opts.numClasses = 21 ;
opts.ignoreXBoundaryBoxes = false ;
opts.addNegativeTargets = false;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

batchSize = size(x, 4) ;

% output the decoded class predictions and extended labels
if isempty(dzdy)

  cellfun(@(x) assert(all(all(x(:,3:4) - x(:,1:2) > 0)), ...
    'MULTIBOXCODER:invalidGroundTruthBoxes', ...
    'ground truth boxes must be in the (xmin, ymin, xmax, ymax) format'), ...
    gt) ;

  % ----------------------------------------------------
  % Decoding predctions, priors and ground truth annotations
  % ----------------------------------------------------

  numGtBoxes = cellfun(@(x) size(x, 1), gt) ;
  locPreds = permute(reshape(x, 4, [], 1, batchSize), [ 2 1 3 4]) ;
  confPreds = permute(reshape(v, opts.numClasses, [], 1, batchSize), ...
                                                          [2 1 3 4]) ;

  % loop over batch
  for i = 1:batchSize

    l_ = l{i} ;
    gt_ = gt{i} ;
    matches = m(i) ;
    hardNegs = n{i} ;
    numGtBoxes_ = numGtBoxes(i) ;
    locPreds_ = locPreds(:,:,:,i) ;
    confPreds_ = confPreds(:,:,:,i) ;

    targetPreds_     = cellfun(@(x) locPreds_(x,:), matches.idx, 'Uni', false) ;
    targetPreds{i}   = vertcat(targetPreds_{:}) ;  
    
    % compute the predicted labels
    posPreds = cellfun(@(x) confPreds_(x,:), matches.idx, 'Uni', false) ;
    negPreds = confPreds_(hardNegs,:) ;
    classPreds_ = vertcat(posPreds{:}, negPreds) ;
    classPreds{i} = permute(classPreds_, [ 3 4 2 1]) ; % prepend singletons

  end

  % append loc predictions for hard negs
  if opts.addNegativeTargets
    for i = 1:batchSize
      targetPreds{end+1} = locPreds(n{i},:,:,i) ;
    end
  end

  % concatenate across the batch
  targetPreds = cat(1, targetPreds{:}) ;
  classPreds = cat(4, classPreds{:}) ;

  numPos = sum(cellfun(@numel, [matches.idx])) ;
 
  % compute weights
  confWeights = ones(1, 1, 1, size(classPreds, 4)) * (1 / sum(numPos)) ;

  if opts.addNegativeTargets
    tWeights = zeros(size(targetPreds)) ;
    tWeights(1:numPos,:,:,:) = (1 / sum(numPos));
  end


else
  dzdLoc = gather(dzdy{1}) ;
  dzdConf = gather(squeeze(dzdy{2})') ;
  numPriors = size(x, 3) / 4 ;

  locDer = zeros(numPriors, 4, 1, batchSize, 'single') ;
  confDer = zeros(numPriors, opts.numClasses, 1, batchSize, 'single') ;

  % reshape the derivatives into the appropriate batch elements
  pos = arrayfun(@(x) {horzcat(m(x).idx{:})}, 1:numel(m)) ;
  neg = cellfun(@(x) {gather(x)}, n) ;

  locDerSizes = cellfun(@numel, pos) ;
  confDerSizes = locDerSizes + cellfun(@numel, neg) ;

  locCumSizes = [ 0 cumsum(locDerSizes) ] ;
  confCumSizes = [ 0 cumsum(confDerSizes) ] ;
  

  for i = 1:batchSize
    locDer_{i} = dzdLoc(locCumSizes(i) + 1:locCumSizes(i+1),:) ;
    confDer_{i} = dzdConf(confCumSizes(i) + 1:confCumSizes(i+1),:) ;
  end

  % loop over batch to fill in the derivatives
  for i = 1:batchSize

    % scale the derivatives according the number of matched priors
    matchPos = pos{i} ;
    matchNeg = neg{i} ;
    matchAll = horzcat(matchPos, matchNeg) ;

    locDerTmp = double(locDer_{i}) ;
    confDerTmp = double(confDer_{i}) ;
 
    sparsePos = sparse(matchPos, 1:numel(matchPos), 1, ...
                        numPriors, numel(matchPos)) ;
    tmp = sparsePos * locDerTmp ;
    locDer(:,:,:,i) = single(tmp) ;
 
    sparseAll = sparse(matchAll, 1:numel(matchAll), 1, ...
                        numPriors, numel(matchAll)) ;
    tmp = sparseAll * confDerTmp ;
    confDer(:,:,:,i) = single(tmp) ;
  end

  locDer = reshape(permute(locDer, [2 1 3 4]), size(x)) ;
  confDer = reshape(permute(confDer, [2 1 3 4]), size(v)) ;

  % return derivatives
  targetPreds = cast(locDer, 'like', x) ; 
  classPreds = cast(confDer, 'like', v) ; 

end
