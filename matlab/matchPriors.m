function matches = matchPriors(gtBoxes, pBoxes, varargin)
% MATCHPRIORS - match prior boxes against gt boxes 
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.overlapThreshold = 0.5 ;
  opts.ignoreXBoundaryBoxes = false ;
  opts.matchRanker = 'overlap' ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  if opts.ignoreXBoundaryBoxes 
    boundaryBoxes = pBoxes(:,1) < 0 ...
                  | pBoxes(:,2) < 0 ...
                  | pBoxes(:,3) > 1 ...
                  | pBoxes(:,4) > 1 ;
    ignored = find(boundaryBoxes) ;
    pBoxes = updateBoundaryPreds(pBoxes, ignored) ;
  else
    ignored = zeros(size(pBoxes, 1), 1) ;
  end

  % re-encode bboxes to [xmin ymin width height]
  gtMinWH = bboxCoder(gtBoxes, 'MinMax', 'MinWH') ;
  pMinWH  = bboxCoder(pBoxes, 'MinMax', 'MinWH') ;
  % first the best overlapping priors with gt boxes
  overlaps = bboxOverlapRatio(gtMinWH, pMinWH, 'Union') ;
  [~, bestOverlapIdx] = max(overlaps, [], 2) ;

  % find any remaining priors with greater than the overlap threshold
  boxIdx = 1:size(gtBoxes, 1) ;
  allMatches_ = arrayfun(@(x, i) ...
                 unique([x find(overlaps(i, :) > opts.overlapThreshold)]), ...
                 bestOverlapIdx', boxIdx, ...
                 'UniformOutput', false) ;

  switch opts.matchRanker
    case 'overlap'
      uniqueMatches = unique(horzcat(allMatches_{:}))' ;
      uniqueMatchOverlaps = overlaps(:,uniqueMatches)' ;
      [~, assignments] = max(uniqueMatchOverlaps, [], 2) ;
    otherwise
      error('%s not recognised', opts.matchRanker) ;
  end

  % Ignore following comment, no longer true...

  % Each prior box that has matched needs to be assigned to a specific
  % ground truth. This process is done in two steps.
  % 1. Initially, all priors are assigned to the gts with which 
  %    they have greatest overlap
  % 2. Then (to ensure that each gt is assigned at least one 
  %    prior) each gt is assigned the prior box which has the
  %    highest overlap with it.  If multiple gts have their 
  %    highest overlap with the same prior, it is assigned
  %    to the gt with which it has greatest overlap and 
  %    the other gts get their 'second choice' of prior etc.

  for i = 1:size(gtBoxes, 1)
      [r,c] = maxMatrixElement(uniqueMatchOverlaps) ;
      assignments(r) = c ;
      uniqueMatchOverlaps(r,:) = -Inf ;
      uniqueMatchOverlaps(:,c) = -Inf ;
  end

  idx = arrayfun(@(x) uniqueMatches(x == assignments)', ...
                  1:size(gtBoxes), 'Uni', false) ;

  matches.idx = idx ;
  matches.ignored = ignored ;
  matches.overlaps = overlaps ;

% ----------------------------------------------------
function pBoxes = updateBoundaryPreds(pBoxes, ignored) 
% ---------------------------------------------------
% replace predictions that lie on the boundary 
% with boxes that cannot be matched by priors

  % choose an unreachable value
  x = max(pBoxes(:)) + 1 ; 

  % update the predictions
  pBoxes(ignored,:) = repmat([ x x x+1 x+1 ], numel(ignored), 1) ;

% ---------------------------------------------
function [row, col] = maxMatrixElement(matrix)
% ---------------------------------------------
  [~,ind] = max(matrix(:)) ;
  [row,col] = ind2sub(size(matrix),ind) ;
