function [targetPreds, classPreds] = vl_nnmultiboxcoder(x, v, m, n, varargin) 
%VL_NNMULTIBOXCODER encodes/decodes labels/predictions into locations
%   [TARGETPREDS, CLASSPREDS] = VL_NNMULTIBOXCODER(X, V, M, N, GT, L) encodes 
%   the training annotations and the network outputs into a common format 
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
%   batch size and C1 = 4 * B.  The input V is a 1 x 1 x C2 x N  
%   array containing the per-class confidence predictions of the network 
%   where C2 = numClasses * B.
%   
%   M is a struct array describing the priors which have been matched against
%   ground truth annotations.  This struct has three fields:
%     M.IDX - The index of the gt box being matched
%     M.IGNORED - A flag indicating whether or not the match should be ignored
%     M.OVERLAP - The overlap between the matched prior and the gt box
%
%   N is a 1 x N cell array of "hard negative" indices. These are the indices
%   of prior boxes which should be predicted as negatives by the netork.
%
%   The output consists of:
%     TARGETPREDS - a M x 4 array containing the encoded network target 
%       predictions.
%     CLASSPREDS - a 1 x 1 x numClasses x K array containing the class 
%       predictions. 
%     
%   [DX, DV] = VL_NNMULTIBOXENCODER(X, V, M, N, DY) computes the 
%   derivatives ot the multibox encoder. DX and DV have the same
%   dimensions as X and V. 
%
%   VL_NNMULTIBOXENCODER(...,'OPT',VALUE,...) takes the following options:
%
%   `numClasses`:: 21
%    The number of classes predicted by the network (used to decode the
%    inputs X and V). 
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.numClasses = 21 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  batchSize = size(x, 4) ;

  % output the decoded class predictions and extended labels
  if isempty(dzdy)

    % Decoding predctions, priors and ground truth annotations
    locPreds = permute(reshape(x, 4, [], 1, batchSize), [ 2 1 3 4]) ;
    confPreds = permute(reshape(v, opts.numClasses, [], 1, batchSize), ...
                                                            [2 1 3 4]) ;

    % loop over batch
    for i = 1:batchSize

      matches = m(i) ;
      hardNegs = n{i} ;
      locPreds_ = locPreds(:,:,:,i) ;
      confPreds_ = confPreds(:,:,:,i) ;

      targetPreds_ = cellfun(@(x) locPreds_(x,:), matches.idx, 'Uni', false) ;
      targetPreds{i}   = vertcat(targetPreds_{:}) ;  %#ok

      % compute the predicted labels
      posPreds = cellfun(@(x) confPreds_(x,:), matches.idx, 'Uni', false) ;
      negPreds = confPreds_(hardNegs,:) ;
      classPreds_ = vertcat(posPreds{:}, negPreds) ;
      classPreds{i} = permute(classPreds_, [ 3 4 2 1]) ; %#ok prepend singletons
    end

    % concatenate across the batch
    targetPreds = cat(1, targetPreds{:}) ;
    classPreds = cat(4, classPreds{:}) ;
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
      locDer_{i} = dzdLoc(locCumSizes(i) + 1:locCumSizes(i+1),:) ; %#ok
      confDer_{i} = dzdConf(confCumSizes(i) + 1:confCumSizes(i+1),:) ; %#ok
    end
    for i = 1:batchSize % loop over batch to fill in the derivatives
      % scale the derivatives according the number of matched priors
      matchPos = pos{i} ; matchNeg = neg{i} ;
      matchAll = horzcat(matchPos, matchNeg) ;
      locDerTmp = double(locDer_{i}) ; confDerTmp = double(confDer_{i}) ;
      sparsePos = sparse(matchPos, 1:numel(matchPos), 1, ...
                          numPriors, numel(matchPos)) ;
      tmp = sparsePos * locDerTmp ; locDer(:,:,:,i) = single(tmp) ;
      sparseAll = sparse(matchAll, 1:numel(matchAll), 1, ...
                          numPriors, numel(matchAll)) ;
      tmp = sparseAll * confDerTmp ; confDer(:,:,:,i) = single(tmp) ;
    end

    locDer = reshape(permute(locDer, [2 1 3 4]), size(x)) ;
    confDer = reshape(permute(confDer, [2 1 3 4]), size(v)) ;

    % return derivatives
    targetPreds = cast(locDer, 'like', x) ; 
    classPreds = cast(confDer, 'like', v) ; 
  end
