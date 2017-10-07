function matches = matchPriors(gtBoxes, pBoxes, l, varargin)
% MATCHPRIORS
%
% TODO: docs
%
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

if false % potentialy faster
  overlaps = vl_nnoverlap(gtBoxes',pBoxes',[]); 
else 
  % re-encode bboxes to [xmin ymin width height]
  gtMinWH = bboxCoder(gtBoxes, 'MinMax', 'MinWH') ;
  pMinWH  = bboxCoder(pBoxes, 'MinMax', 'MinWH') ;
  % first the best overlapping priors with gt boxes
  overlaps = bboxOverlapRatio(gtMinWH, pMinWH, 'Union') ;
end
[~, bestOverlapIdx] = max(overlaps, [], 2) ;

% find any remaining priors with greater than the overlap threshold
boxIdx = 1:size(gtBoxes, 1) ;
allMatches_ = arrayfun(@(x, i) ...
               unique([x find(overlaps(i, :) > opts.overlapThreshold)]), ...
               bestOverlapIdx', boxIdx, ...
               'UniformOutput', false) ;

switch opts.matchRanker
  case 'size'
    % match from smallest to largest gt
    areas = prod(gtMinWH(:,[3 4]), 2) ;
    [~, sIdx] = sort(areas, 'ascend') ;
    gtMinWH
    keyboard ;
  case 'overlap'
    matchSizes = cellfun(@numel, allMatches_) ;
    uniqueMatches = unique(horzcat(allMatches_{:}))' ;
    uniqueMatchOverlaps = overlaps(:,uniqueMatches)' ;
    [~, assignments] = max(uniqueMatchOverlaps, [], 2) ;
  otherwise
    error(sprintf('%s not recognised', opts.matchRanker)) ;
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

if 0  
  % re-run matching until all gt objects have been matched
  missing = cellfun(@isempty, idx) ;
  repeat = find(missing) ;
  if ~isempty(repeat)

    classes = {'none_of_the_above','aeroplane', 'bicycle', 'bird', 'boat', ...
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', ...
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' } ;

    path = fullfile(vl_rootnn, 'data/matcher', ...
          sprintf('dropped-%.1f-overlap.txt', opts.overlapThreshold)) ;

    fid = fopen(path, 'a') ;
    for i = 1:numel(repeat)
      fprintf('dropped label %s\n', classes{l(repeat(i))}) ;
      fprintf(fid, '%d\n', l(repeat(i))) ;
    end
    fclose(fid) ;
    %gtMinWH_ = gtMinWH(missing,:) ;
    %gtMinWH_ = gtMinWH(missing,:) ;
  end
end

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
[maxVal,ind] = max(matrix(:)) ;
[row,col] = ind2sub(size(matrix),ind) ;
