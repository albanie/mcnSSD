classdef utrunpatchtrials < matlab.unittest.TestCase

  properties (TestParameter)
    opts = struct('numTrials', 50,  ...
                  'minAspect', 0.5, ...
                  'maxAspect', 2, ...
                  'minPatchScale', 0.1, ...
                  'maxPatchScale', 1) ;
  end

  methods (Test)

    function checkRandSourceSizeAssertion(test)
      opts = test.opts ;
      randSource = rand(opts.numTrials + 1, 4) ;
      targetsWH = [] ;
      strategy = 'rand_patch' ;
      test.verifyError(@() ...
        runPatchTrials(targetsWH, strategy, randSource, opts), ...
        'PATCHTRIALS:incorrectRandSize') ;
    end

    function checkRandSourceRangeAssertion(test)
      opts = test.opts ;
      randSource = ones(opts.numTrials, 4) * 1.1;
      targetsWH = [] ;
      strategy = 'rand_patch' ;
      test.verifyError(@() ...
        runPatchTrials(targetsWH, strategy, randSource, opts), ...
        'PATCHTRIALS:incorrectRandRange') ;
    end

    function checkPatchSize(test)
      opts = test.opts ;
      randSource = ones(opts.numTrials, 4) * 0.5;
      targetsWH = [ 0.1 0.1 0.9 0.9 ] ;
      strategy = 'rand_patch' ;
      patch = runPatchTrials(targetsWH, strategy, randSource, opts) ;
      expectedPatchSize = 0.3025 ;
      patchSize = (patch(3) - patch(1)) * (patch(4) - patch(2)) ;
      test.verifyEqual(patchSize, expectedPatchSize, 'AbsTol', 1e-10) ;
    end

    function checkNoMatchesReturnsOriginalImg(test)
      opts = test.opts ;
      randSource = ones(opts.numTrials, 4) * 0.5;
      targetsWH = [ 0.1 0.1 0.15 0.15 ] ;
      strategy = 'jacc_0.9' ;
      patch = runPatchTrials(targetsWH, strategy, randSource, opts) ;
      expectedPatch = [] ;
      test.verifyEqual(patch, expectedPatch, 'AbsTol', 1e-10) ;
    end
  end
end
