classdef utpatchsampler < matlab.unittest.TestCase

  properties (TestParameter)
    opts = struct('numTrials', 50,  ...
                  'minPatchSize', 0.3, ...
                  'minAspect', 0.5, ...
                  'maxAspect', 2) ;
  end

  methods (Test)

    function checkPatchDistribution(test)
      % sanity check on the patch sampler
      opts = test.opts ;
      originalImg = [ 0 0 1 1 ] ;
      targets = [0 0 sqrt(0.5) sqrt(0.5)] ;
      labels = [1] ;
      numSimulations = 1000 ;
      patches = zeros(numSimulations, 4) ;

      % count the number of patches that return the
      % original image
      count = 0 ;
      for i = 1:numSimulations
          [patch,~,~] = patchSampler(targets, labels, opts) ;
          if all(patch == originalImg)
              count = count + 1 ;
          end
      end
      expectedBounds = [0.3 * numSimulations, 0.4 * numSimulations] ;
      inRange = expectedBounds(1) < count && count < expectedBounds(2) ;
      test.verifyTrue(inRange) ;
    end
  end
end
