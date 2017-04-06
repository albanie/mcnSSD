function hardNegs = hardNegatives(preds, matches, allOverlaps, ignored, varargin)
%TODO DOCS

opts.negPosRatio = 3 ;
opts.backgroundLabel = 1 ;
opts.negOverlap = 0.5 ;
opts.numClasses = 21 ;
opts.ignoreXBoundaryBoxes = false ;
opts = vl_argparse(opts, varargin) ;

% determine the number of neg samples required
numSamples = size(preds, 1) ;
numPos = sum(cellfun(@numel, matches)) ;
numNeg = min(opts.negPosRatio * numPos, numSamples - numPos) ;

% compute losses for predictions
% NOTE: To make the computation a little smoother we assume that 
% the correct class for all of these predictions is `background`
% then remove the loss for the predictions whose label was 
% actually foreground
maxPreds = max(preds, [], 2) ;
ep = exp(bsxfun(@minus, preds, maxPreds)) ;
loss = maxPreds + log(sum(ep, 2)) - preds(:, opts.backgroundLabel) ;

% remove maxProbs with matched priors
loss(unique(horzcat(matches{:}))) = -Inf ;

% if required, ignore boundary boxes
if opts.ignoreXBoundaryBoxes
    loss(ignored) = -Inf ;
end

[~, rankedIdx] = sort(loss, 'descend') ;
hardNegs = rankedIdx(1:numNeg) ;
