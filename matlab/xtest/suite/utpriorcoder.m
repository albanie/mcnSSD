classdef utpriorcoder < matlab.unittest.TestCase
  methods (Test)

    function checkInverse(test)
      batchSize = 1 ;
      numPriors = 3 ;

      vars = [0.1 0.1 0.2 0.2] ;
      pBoxes = [ 0.4 0.2 0.7 0.7 ;
                 0.1 0.1 0.4 0.41 ;
                 0.26 0.26 0.75 0.75 ]  ;
      pVars = repmat(vars, [numPriors 1]) ;

      x = rand(numPriors,4, 1, 1) ;

      y = priorCoder(x, pBoxes, pVars, 'targets') ;
      y_inv = priorCoder(y, pBoxes, pVars, 'CenWH') ;
      test.verifyEqual(x, y_inv, 'AbsTol', 1e-5) ;
    end

  end
end
