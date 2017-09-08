classdef utpatchsampler < matlab.unittest.TestCase

  properties (TestParameter)
    opts = struct('numTrials', 50,  ...
                  'minAspect', 0.5, ...
                  'maxAspect', 2, ...
                  'minPatchScale', 0.3, ...
                  'maxPatchScale', 1, ...
                  'clipTargets', 0) ;
  end

  methods (Test)

    function checkPatchDistribution(test)
      % sanity check on the patch sampler
      originalImg = [ 0 0 1 1 ] ;
      targets = [0 0 sqrt(0.5) sqrt(0.5)] ;
      numSimulations = 1000 ;
      labels = 1 ;

      % count the number of patches that return the
      % original image
      count = 0 ;
      for i = 1:numSimulations
          [patch,~,~] = patchSampler(targets, labels, test.opts) ;
          if all(patch == originalImg)
              count = count + 1 ;
          end
      end
      expectedBounds = [0.15 * numSimulations, 0.3 * numSimulations] ;
      inRange = expectedBounds(1) < count && count < expectedBounds(2) ;
      test.verifyTrue(inRange) ;
    end
  end
end
