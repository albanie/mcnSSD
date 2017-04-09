classdef nnmultiboxcoder < nntest
  methods (Test)

    function basic(test)
        batchSize = 5 ;
        numPriors = 7 ;
        numLocPreds = numPriors * 4 ;
        numConfPreds = numPriors * 21 ;
        labelRange = [1 21] ;
        for i = 1:batchSize
            numBoxes = randi(10, 1) ;
            xmin = rand(numBoxes, 1) ;
            ymin = rand(numBoxes, 1) ;
            xmax = xmin + (1 - xmin) .* rand(numBoxes, 1) ;
            ymax = ymin + (1 - ymin) .* rand(numBoxes, 1) ;
            gt{i} = [ xmin ymin xmax ymax] ;
            l{i} = randi(21, 1, numBoxes) ;
        end

        % we need to encode some viable prior boxes to pass through 
        % the saftey checks
        vars = [0.1 0.1 0.2 0.2] ;
        priorBoxes = [ 0.4 0.2 0.7 0.7 ;
                     0.1 0.1 0.4 0.41 ;
                     0.1 0.1 0.8 0.8 ;
                     0.1 0.1 0.8 0.9 ;
                     0.1 0.1 0.9 0.8 ;
                     0.2 0.1 0.9 0.8 ;
                     0.26 0.26 0.75 0.75 ]'  ;
        priorVars = repmat(vars', [numPriors 1]) ;

        x = randn(1,1,numLocPreds, batchSize,'single') ;
        v = randn(1,1,numConfPreds, batchSize,'single') ;
        p = cat(3, priorBoxes(:), priorVars) ;

        y = vl_nnmultiboxcoder(x, v, p, gt, l) ;

        posMatches = y{5} ;
        negMatches = y{6} ;

        %check derivatives with numerical approximation
        derLocs = test.randn(size(y{1})) / 100 ;
        derConfs = test.randn(size(y{3})) / 100 ;
        dzdy = { derLocs, derConfs } ;
        dzdxv = vl_nnmultiboxcoder(x, v, p, gt, l, dzdy, ...
                                    'matchingPosIndices', posMatches, ...
                                    'matchingNegIndices', negMatches) ;
        dzdx = dzdxv{1} ;
        dzdv = dzdxv{2} ;

        test.der(@(x) forward_wrapper(x,v,p,gt,l,'locPreds'), x, ...
                                         dzdy{1}, dzdx, test.range * 1e-3) ;

        % NOTE: the delta on this test must be small to avoid "flipping" the 
        % hard negatives
        test.der(@(v) forward_wrapper(x,v,p,gt,l,'confPreds'), v, ...
                                          dzdy{2}, dzdv, test.range * 1e-5) ;
    end

  end
end

% ------------------------------------------------------
function y = forward_wrapper(x,v,p,gt,l,target)
% ------------------------------------------------------

y = vl_nnmultiboxcoder(x,v,p,gt,l) ;
switch target
    case 'locPreds'
        y = y{1} ;
    case 'confPreds'
        y = y{3} ;
    otherwise
        error('unrecognized derivative') ;
    end
end
