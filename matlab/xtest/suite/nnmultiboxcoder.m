classdef nnmultiboxcoder < nntest
  methods (Test)

    function basic(test)
      batchSize = 5 ;
      numPriors = 40 ;
      numLocPreds = numPriors * 4 ;
      numConfPreds = numPriors * 21 ;
      m = struct() ; n = cell(1, batchSize) ;
      for ii = 1:batchSize
        numBoxes = randi(10, 1) ;
        m(ii).idx = num2cell(randsample(numPriors, numBoxes)) ;
        m(ii).ignored = false ;
        n{ii} = randsample(setdiff(1:numPriors, [m(ii).idx{:}]), numBoxes * 3) ;
      end

      x = randn(1,1,numLocPreds, batchSize,'single') ;
      v = randn(1,1,numConfPreds, batchSize,'single') ;

      [targetPreds, classPreds] = vl_nnmultiboxcoder(x, v, m, n) ;

      %check derivatives with numerical approximation
      derLocs = test.randn(size(targetPreds)) / 100 ;
      derConfs = test.randn(size(classPreds)) / 100 ;
      [derLoc_, derConf_] = vl_nnmultiboxcoder(x, v, m, n, derLocs, derConfs) ;
      test.der(@(x) forward_wrapper(x, v, m, n,'locPreds'), x, ...
                                    derLocs, derLoc_, test.range * 1e-3) ;

      % NOTE: the delta on this test must be small to avoid "flipping" the
      % hard negatives
      test.der(@(v) forward_wrapper(x,v,m,n,'confPreds'), v, ...
                                   derConfs, derConf_, test.range * 1e-5) ;
    end

  end
end

% ------------------------------------------------------
function y = forward_wrapper(x,v,m,n, target)
% ------------------------------------------------------
  [locs, confs] = vl_nnmultiboxcoder(x, v, m, n) ;
  switch target
    case 'locPreds', y = locs ;
    case 'confPreds', y = confs ;
    otherwise, error('unrecognized derivative') ;
  end
end
