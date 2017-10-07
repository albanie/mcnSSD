function [matches, targets, tWeights, boxes] = vl_nnmatchpriors(p, gt, l, varargin) 
%VL_NNMATCHPRIORS matches prior boxes against ground truth annotations
%   Y = VL_NNMATCHPRIORS(P, GT, l, []) encodes the training
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
%   The input GT is a 1 x 1 x C5 x N array containing the 
%   annotations 
%   
%   The output y is a 1 x 6 cell array whose values are the following:
%   outputs{1} is a vector of ground truth bounding box targets 
%   outputs{2} is a vector of matched positive prior box indices
%
%   VL_NNMATCHPRIORS(...,'OPT',VALUE,...) takes the following options:
%
%   `overlapThreshold`:: 0.5 
%    The threshold used to determine whether a ground truth annotation is 
%    to be matched to a given prior box. The prior box then becomes a 
%    positive example during training.

opts.overlapThreshold = 0.5 ;
opts.ignoreXBoundaryBoxes = false ;
opts.maxScore = false;
opts.normAG = 0 ;
opts.normAGpop = 0 ;
opts.normAGpopComb = 0 ;
opts.matchRanker = 'overlap' ;
opts.fusedConfs = [];
opts.matchesIn = [];
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

batchSize = numel(gt) ;

cellfun(@(x) assert(all(all(x(:,3:4) - x(:,1:2) > 0)), ...
        'MULTIBOXCODER:invalidGroundTruthBoxes', ...
        'ground truth boxes must be in the (xmin, ymin, xmax, ymax) format'), ...
        gt) ;

numGtBoxes = cellfun(@(x) size(x, 1), gt) ;
numPriors = size(p, 1) / 4 ;

% output the decoded class predictions and extended labels
assert(isempty(dzdy), 'prior matching is not performed on the back pass') ;

% --------------------------------------------
% Decoding priors and ground truth annotations
% --------------------------------------------
% Priors are identical across a batch, so we only produce one set per batch
priors = reshape(p, numPriors, 4, 2, size(p,4)) ;
pBoxes = permute(reshape(priors(:,:,1,:), 4, [], 1, size(p,4) ),[2 1 3 4]) ;
pVar   = permute(reshape(priors(:,:,2,:), 4, [], 1, size(p,4) ),[2 1 3 4]) ;

constantPriors = size(pBoxes,4) == 1;

pBoxes = gather(pBoxes) ;
pCenWH = bboxCoder(pBoxes, 'MinMax', 'CenWH') ;

batchIdxCell = mat2cell(1:batchSize,1,ones(1,batchSize));
if ~isempty(opts.matchesIn)
    matches = opts.matchesIn;
else
    if constantPriors
        matches = cellfun(@(x,y) matchPriors(x, pBoxes, y, ...
                     'overlapThreshold', opts.overlapThreshold, ...
                     'matchRanker', opts.matchRanker, ...
                     'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes), gt, l) ;
    else
        matches = cellfun(@(x,imidx,y) matchPriors(x, pBoxes(:,:,:,imidx), y, ...
                     'overlapThreshold', opts.overlapThreshold, ...
                     'matchRanker', opts.matchRanker, ...
                     'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes), gt, batchIdxCell, l) ;
    end
end

% Repeat each gt bounding box for every prior it has been matched against
% (enables vectorization of the bbox target computation)
gtCenWH = cellfun(@(x) {bboxCoder(x, 'MinMax', 'CenWH')}, gt) ;
matchIdx = arrayfun(@(x) {horzcat(matches(x).idx{:})}, 1:batchSize) ;
matchesPerGt = arrayfun(@(x) {cellfun(@numel, matches(x).idx)'}, 1:batchSize) ;
repGtBoxesWH = arrayfun(@(x) {repeatBoxes(gtCenWH{x}, matchesPerGt{x})}, ...
                              1:batchSize) ;


if constantPriors
    pVar = cellfun(@(x) {pVar(x, :)}, matchIdx) ;
    pCenWH = cellfun(@(x) {pCenWH(x, :)}, matchIdx) ;
else 
    pVar = cellfun(@(x,bidx) {pVar(x, :, :, bidx)}, matchIdx, batchIdxCell) ;
    pCenWH = cellfun(@(x,bidx) {pCenWH(x, :,:,bidx )}, matchIdx,batchIdxCell) ;
end

% fuse outputs
gtBoxes = vertcat(repGtBoxesWH{:}) ;
pBoxes = vertcat(pCenWH{:}) ;
pVars = vertcat(pVar{:}) ;
boxes = {gtBoxes, pBoxes, pVars} ;

% Compute target offsets across the batch
targets = priorCoder(gtBoxes, pBoxes, pVars, 'targets') ;

% create a weight for each target
tWeights = ones(size(targets)) ;

if opts.normAG
  [include, counts, extendedLabels] = labelsInBatch(matches, l) ;
  w = mean(counts) ./ counts ;

  for i = 1:numel(w)
    label = include(i) ;
    tWeights(extendedLabels == label,:) = w(i) ;
  end

elseif opts.normAGpop % population weights
  [include, counts, extendedLabels] = labelsInBatch(matches, l) ;

  % it wouldn't be a proper project without some hardcoded constants
  % (derived from VOC population statistics)
   w = [ ...
   1.1341, 1.2064, 0.8007, 1.0432, 0.6887, 1.6033, 0.3636, 0.9018, 0.3360, ...
   1.3775, 1.3788, 0.7010, 1.2607, 1.2773, 0.0936, 0.8453, 1.0819, 1.2034, ...
   1.4811, 1.2216
   ] ;
  for i = 1:numel(include)
    label = include(i) ;
    tWeights(extendedLabels == label,:) = w(label-1) ;
  end
elseif opts.normAGpopComb

  [include, counts, extendedLabels] = labelsInBatch(matches, l) ;
  [includeRaw, countsRaw] = labelCounts([l{:}]) ;

  % occasionally, labels will be dropped by the matcher so we must
  % handle this case
  keep = find(ismember(includeRaw, include)) ;
  includeRaw = includeRaw(keep) ;
  countsRaw = countsRaw(keep) ;

  % (derived from VOC population statistics)
  % wPriors = [ ...
      %0.0272, 0.0256, 0.0385, 0.0296, 0.0448, 0.0192, 0.0849, 0.0342, 0.0919, ...
      %0.0224, 0.0224, 0.0440, 0.0245, 0.0242, 0.3298, 0.0365, 0.0285, 0.0256, ...
      %0.0208, 0.0253 ] ;

   % scaled inverse priors
   wInvPriors = [ ...
    1.1343, 1.2052, 0.8014, 1.0424, 0.6887, 1.6070, 0.3634, 0.9022, ...
    0.3357, 1.3774, 1.3774, 0.7012, 1.2593, 1.2749, 0.0936, 0.8453, ...
    1.0826, 1.2052, 1.4833, 1.2195 ...
    ] ;

  cW = countsRaw ./ counts ;
  qW = wInvPriors(includeRaw - 1) ;

  % rebalance for ratio of raw to extended labels 
  for i = 1:numel(include)
    % offset for background
    label = include(i) ; 
    tWeights(extendedLabels == label, :) = cW(i) * qW(i) ;
  end
end

% scale weights
tWeights = tWeights * (1 / size(targets, 1)) ;

if opts.maxScore
	error('not done yet');
	nCls      = size(opts.fusedConfs,3)/numPriors; assert(round(nCls)==nCls);
	batchSize = size(opts.fusedConfs,4);
	confPreds = reshape(opts.fusedConfs, nCls, numPriors, batchSize) ;
end

% ------------------------------------------
function boxes = repeatBoxes(boxes, repeats)
% ------------------------------------------
boxes = arrayfun(@(x) {repmat(boxes(x,:), [repeats(x) 1])}, 1:numel(repeats)) ;
boxes = vertcat(boxes{:}) ;

% --------------------------------------------------------------------
function [include, counts, extendedLabels] = labelsInBatch(matches, l) 
% --------------------------------------------------------------------
for i = 1:numel(matches)
  l_ = l{i} ;
  m_ = matches(i).idx ;
  repL_ = arrayfun(@(x,y) {repmat(x,1,y)}, l_, cellfun(@numel, m_)) ;
  repL{i} = [repL_{:}] ;
end
extendedLabels = [repL{:}] ;
include = unique(extendedLabels) ;

% handle the single element case 
if numel(include) == 1
  counts = 1 ; labels = include ;
else
  [counts,labels] = hist(extendedLabels, include) ;
end

% ----------------------------------------------
function [include, counts] = labelCounts(rawLabels) 
% ----------------------------------------------

include = unique(rawLabels) ;

% handle the single element case 
if numel(include) == 1
  counts = 1 ; labels = include ; w = [] ;
else
  [counts, labels] = hist(rawLabels, include) ;
  w = mean(counts(2:end)) ./ counts(2:end) ;
end
