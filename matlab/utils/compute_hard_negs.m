function hardNegs = compute_hard_negs(preds, matches, varargin)
% COMPUTE_HARD_NEGS - hard negative mining for prior boxes 
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.negPosRatio = 3 ;
  opts.backgroundLabel = 1 ;
  opts.negOverlap = 0.5 ;
  opts.numClasses = 21 ;
  opts.ignoreXBoundaryBoxes = false ;
  opts = vl_argparse(opts, varargin) ;

  % determine the number of neg samples required
  numSamples = size(preds, 1) ;
  numPos = sum(cellfun(@numel, matches.idx)) ;
  numNeg = min(round(opts.negPosRatio * numPos), numSamples - numPos) ;

  % compute losses for predictions
  % NOTE: To make the computation a little smoother we assume that 
  % the correct class for all of these predictions is `background`
  % then remove the loss for the predictions whose label was 
  % actually foreground
  maxPreds = max(preds, [], 2) ;
  ep = exp(bsxfun(@minus, preds, maxPreds)) ;
  loss = maxPreds + log(sum(ep, 2)) - preds(:, opts.backgroundLabel) ;

  % remove maxProbs with matched priors
  loss(unique(horzcat(matches.idx{:}))) = -Inf ;

  % if required, ignore boundary boxes
  if opts.ignoreXBoundaryBoxes
      loss(matches.ignored) = -Inf ;
  end 

  [~, rankedIdx] = sort(gather(loss), 'descend') ;
  hardNegs = rankedIdx(1:numNeg) ;
  if isa(preds,'gpuArray'), hardNegs = gpuArray(hardNegs) ; end
