function y = vl_nnmultiboxcoder(x, v, p, gt, l, dzdy, varargin) 
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
%   The input P is a C3 x 1 x 2 x N array containing the prior boxes, 
%   which are encoded as a set of bbox coordinates (xminx, ymin, 
%   xmax, ymax) and a set of four "variances" which are used to scale 
%   the resulting boxes, where C3 = 4 * B
%   
%   The input GT is a 1 x 1 x C5 x N array containing the 
%   annotations .....................TODO

%   The input L is a 1 x 1 x C4 x N array containing the 
%   annotations .....................TODO
%
%   TODO: docuemnt the outputs properly....
%   
%   The output y is a 1 x 6 cell array whose values are the following:
%   outputs{1} is a vector of encoded location predictions
%   outputs{2} is a vector of ground truth bounding box targets 
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
%   `locWeight`:: 1
%    A scalar used to weight the location loss (produced by applying the 
%    Huber loss to the bounding box regression) against the class 
%    prediction loss (typically a softmax). 
%
%   `normalization`:: 'VALID'
%    A string denoting the form of normalization applied to loss outputs
%    returned to this layer. Can be 'NONE', in which case the losses are
%    simply summed at every location, or 'VALID', in which case the loss
%    is divided by the number of contributing inputs (similar to caffe's
%    normalization loss layer parameter) TODO: Not yet implemented
%
%   `shareLocation`:: true 
%    TODO: doc
%    TODO: Not yet implemented
%
%   `overlapThreshold`:: 0.5 
%    The threshold used to determine whether a ground truth annotation is 
%    to be matched to a given prior box. The prior box then becomes a 
%    positive example during training.
%
%   `usePriorForMatching`:: true 
%    TODO: doc
%    TODO: Not yet implemented
%
%   `hardNegativeMining`:: true 
%    If true, following the matching phase the negative prior boxes are 
%    ranked in order of netowrk class confidence for the non-background
%    object classes.  Only the top ranked negative instances are then 
%    used during training.  If false, all negatives are used.
%    TODO: false is not yet implemented
%
%   `hardNegativeMining`:: true 
%    If true, following the matching phase the negative prior boxes are 
%    ranked in order of netowrk class confidence for the non-background
%    object classes.  Only the top ranked negative instances are then 
%    used during training.  If false, all negatives are used.
%
%   `negPosRatio`:: 3 
%    The ratio of negative to positive instances used during training 
%    (note that this option is only used if `hardNegativeMining` is 
%    true).
%
%   `backgroundLabel`:: 1 
%    Labels in `L` with this value are not used as positive training 
%    instances.
%
%   `matchingPosIndices`:: [] 
%    The indices of the positively matched prior boxes (only used in 
%    backward pass.
%
%   `matchingNegIndices`:: [] 
%    The indices of the negatively matched prior boxes (only used in 
%    backward pass.

opts.locWeight = 1 ;
opts.negPosRatio = 3 ;
opts.numClasses = 21 ;
opts.backgroundLabel = 1 ;
opts.shareLocation = true ;
opts.overlapThreshold = 0.5 ;
opts.normalization = 'VALID' ;
opts.usePriorForMatching = true ;
opts.hardNegativeMining = true ;
opts.matchingPosIndices = {} ;
opts.matchingNegIndices = {} ;
opts.ignoreXBoundaryBoxes = false ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

% until larger batch sizes are handled properly
batchSize = size(x, 4) ;

cellfun(@(x) assert(all(all(x(:,3:4) - x(:,1:2) > 0)), ...
        'MULTIBOXCODER:invalidGroundTruthBoxes', ...
        'ground truth boxes must be in the (xmin, ymin, xmax, ymax) format'), ...
        gt) ;

numGtBoxes = cellfun(@(x) size(x, 1), gt) ;
numPriors = size(p, 1) / 4 ;

% output the decoded class predictions and extended labels
if nargin <= 1 || isempty(dzdy)

    % ----------------------------------------------------
    % Decoding predctions, priors and ground truth annotations
    % ----------------------------------------------------

    locPreds = permute(reshape(x, 4, numPriors, 1, batchSize), [ 2 1 3 4]) ;
    confPreds = permute(reshape(v, opts.numClasses, numPriors, 1, batchSize), ...
                                                            [2 1 3 4]) ;

    % Priors are identical across a batch, so we only produce one 
    % set per batch
    priors = reshape(p, numPriors, 4, 2) ;
    pBoxes = reshape(priors(:,:,1), 4, [])' ;
    pVar = reshape(priors(:,:,2), 4, [])' ;

    % loop over batch
    for i = 1:batchSize

        locPreds_ = locPreds(:,:,:,i) ;
        confPreds_ = confPreds(:,:,:,i) ;
        numGtBoxes_ = numGtBoxes(i) ;
        l_ = l{i} ;
        gt_ = gt{i} ;

        % ----------------------------------------------------
        % Matching 
        % ----------------------------------------------------

        pBoxes = gather(pBoxes) ;
        [matches, allOverlaps, ignored] = matchPriors(gt_, pBoxes, ...
                     'overlapThreshold', opts.overlapThreshold, ...
                     'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes) ;
        matchesPerGt = cellfun(@numel, matches)' ;
        matchIdx = horzcat(matches{:}) ;

        % re-format bounding boxes
        gtCenWH = bboxCoder(gt_, 'MinMax', 'CenWH') ;
        pCenWH = bboxCoder(pBoxes, 'MinMax', 'CenWH') ;

        % Repeat the each gt bounding for every prior it has been matched 
        % to box to (allows vectorization of the bbox target computation)
        repGtBoxesWH_ = arrayfun(@(x) repmat(gtCenWH(x,:), ...
                                [matchesPerGt(x) 1]), 1:numGtBoxes_, ...
                                'Uni', false) ;
        repGtBoxesWH = vertcat(repGtBoxesWH_{:}) ;
            
        % ------------------------------------------------------
        % Compute target offsets
        % ------------------------------------------------------
        targets{i} = priorCoder(repGtBoxesWH, pCenWH(matchIdx,:), ...
                        pVar(matchIdx, :), 'targets') ;

        % DEBUG
        if any(any(imag(targets{i})))
            keyboard
        end

        targetPreds_ = cellfun(@(x) locPreds_(x,:), matches, 'Uni', false) ;
        targetPreds{i} = vertcat(targetPreds_{:}) ;

        % Add hard negatives
        hardNegs = hardNegatives(confPreds_, matches, allOverlaps, ignored, ...
                                    'backgroundLabel', opts.backgroundLabel, ...
                                    'negPosRatio', opts.negPosRatio, ...
                                    'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes) ;

        % store the matching indices for backprop pass
        % NOTE: only the positives are used to compute the regression loss
        matchingPosIndices{i} = horzcat(matches{:})' ;
        matchingNegIndices{i} = hardNegs ;

        % repeat labels for each ground truth match
        extendedLabels_ = arrayfun(@(x) repmat(l_(x), [1 numel(matches{x})]), ...
                                    1:numGtBoxes_, 'Uni', false) ;

        % add the negative labels
        extendedLabels{i} = horzcat(extendedLabels_{:}, ...
                                ones(1, numel(hardNegs)) * opts.backgroundLabel) ;

        % compute the predicted labels
        posPreds = cellfun(@(x) confPreds_(x,:), matches, 'Uni', false) ;
        negPreds = confPreds_(hardNegs,:) ;
        classPreds_ = vertcat(posPreds{:}, negPreds) ;
        classPreds{i} = permute(classPreds_, [ 3 4 2 1]) ; % prepend singletons
    end

    % concatenate across the batch
    targetPreds = cat(1, targetPreds{:}) ;
    targets = cat(1, targets{:}) ;
    classPreds = cat(4, classPreds{:}) ;
    extendedLabels = cat(2, extendedLabels{:}) ;

    y = { targetPreds, targets, classPreds, extendedLabels, ...
          matchingPosIndices, matchingNegIndices } ;

else
    % sparse matrix operations are (currently) more efficient on the CPU
    dzdLoc = gather(dzdy{1}) ;
    dzdConf = gather(squeeze(dzdy{2})') ;

    locDer = zeros(numPriors, 4, 1, batchSize, 'single') ;
    confDer = zeros(numPriors, opts.numClasses, 1, batchSize, 'single') ;

    % reshape the derivatives into the appropriate batch elements
    locDerSizes = cellfun(@numel, opts.matchingPosIndices) ;
    confDerSizes = cellfun(@(x,y) numel(x) + numel(y), ...
                            opts.matchingPosIndices, ...
                            opts.matchingNegIndices) ;

    locCumSizes = [ 0 cumsum(locDerSizes) ] ;
    confCumSizes = [ 0 cumsum(confDerSizes) ] ;
    
    for i = 1:batchSize
        locDer_{i} = dzdLoc(locCumSizes(i) + 1:locCumSizes(i+1),:) ;
        confDer_{i} = dzdConf(confCumSizes(i) + 1:confCumSizes(i+1),:) ;
    end

    % loop over batch to fill in the derivatives
    for i = 1:batchSize

        % scale the derivatives according the number of matched priors
        matchPos = gather(opts.matchingPosIndices{i}) ;
        matchNeg = gather(opts.matchingNegIndices{i}) ;
        matchAll = vertcat(matchPos, matchNeg) ;
    
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

    locDer = cast(reshape(permute(locDer, [2 1 3 4]), size(x)), 'like', x) ;
    confDer = cast(reshape(permute(confDer, [2 1 3 4]), size(v)), 'like', v) ;

    y = {locDer, confDer} ;
end
