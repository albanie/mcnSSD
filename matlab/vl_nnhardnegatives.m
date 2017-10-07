function [hardNegs, extendedLabels, cWeights] = ...
                           vl_nnhardnegatives(v, l, m, varargin) 

  opts.numClasses = 21 ; 
  opts.negPosRatio = 3 ;
  opts.backgroundLabel = 1 ;
  opts.ignoreXBoundaryBoxes = false ;
  opts.batchRanking = false ;
  opts.normAG = false ;
  opts.normAGpop = false ;
  opts.normAGpopComb = false ;
  [opts, ~] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  batchSize = size(v, 4) ; 
  numGtBoxes = cellfun(@numel, l) ;
  confPreds = permute(reshape(v, opts.numClasses, [], 1, batchSize), ...
                                                          [2 1 3 4]) ;

  % loop over batch
  hardNegs = cell(1, batchSize) ;
  extendedLabels = cell(1, batchSize) ;
  for i = 1:batchSize

    l_ = l{i} ;
    matches = m(i) ;

    % Add hard negatives
    hardNegs{i} = compute_hard_negs(confPreds(:,:,:,i), matches, ...
                       'backgroundLabel', opts.backgroundLabel, ...
                       'negPosRatio', opts.negPosRatio, ...
                       'ignoreXBoundaryBoxes', opts.ignoreXBoundaryBoxes)' ;

    % repeat labels for each ground truth match 
    extendedLabels_ = arrayfun(@(x) repmat(l_(x), ...
                                [1 numel(matches.idx{x})]), ...
                                1:numGtBoxes(i), 'Uni', false) ;
    % add the negative labels 
    extendedLabels{i} = horzcat(extendedLabels_{:}, ...
                       ones(1, numel(hardNegs{i})) * opts.backgroundLabel) ;
  end

  numPos = sum(cellfun(@numel, [m.idx])) ;
  numNeg = sum(cellfun(@numel, hardNegs)) ;
  extendedLabels = cat(2, extendedLabels{:}) ;

  % create a weight for each value
  cWeights = ones(1, 1, 1, numPos + numNeg) ;

  % scale weights
  cWeights = cWeights * (1 / sum(numPos)) ;
