classdef MultiboxCoder < dagnn.ElementWise
  properties
    numClasses = 21
    overlapThreshold = 0.5
    backgroundLabel = 1
    negPosRatio = 3
    negOverlap = 0.5
    matchingPosIndices = []
    matchingNegIndices = []
  end

  methods
    function outputs = forward(obj, inputs, params)
        y = vl_nnmultiboxcoder(inputs{1}, inputs{2}, inputs{3}, inputs{4}, ...
                                 inputs{5}, ...
                                 'numClasses', obj.numClasses, ...
                                 'overlapThreshold', obj.overlapThreshold, ...
                                 'backgroundLabel', obj.backgroundLabel, ...
                                 'negPosRatio', obj.negPosRatio) ;

        % store the matching indices for the backwards pass
        obj.matchingPosIndices = y{5} ;
        obj.matchingNegIndices = y{6} ;
        numPos = cellfun(@numel, obj.matchingPosIndices) ;
        numNeg = cellfun(@numel, obj.matchingNegIndices) ;

        locWeights = ones(size(y{1})) * (1 / sum(numPos)) ;
        confWeights = ones(1, 1, 1, size(y{3}, 4)) * (1 / sum(numPos)) ;

        y{5} = locWeights ;
        y{6} = confWeights ;
        outputs = y ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

       % drop derivatives from priors, labels and gt boxes
       derOutputs([2 4 5]) = [] ;

       derInputs = vl_nnmultiboxcoder(inputs{1}, inputs{2}, inputs{3}, ...
                                 inputs{4}, inputs{5}, derOutputs, ...
                                 'numClasses', obj.numClasses, ...
                                 'backgroundLabel', obj.backgroundLabel, ...
                                 'negPosRatio', obj.negPosRatio, ...
                                 'matchingPosIndices', obj.matchingPosIndices, ...
                                 'matchingNegIndices', obj.matchingNegIndices) ;
        derInputs(3:5) = cell(1,3) ;
        derParams = {} ;
    end


    function outputSizes = getOutputSizes(obj, inputSizes)
      %outputSizes{1} = inputSizes{1}(obj.order) ;
      keyboard
      outputSizes{1} = [ 20 2 2 1] ;
    end

    function rfs = getReceptiveFields(obj)
      rfs = {} ;
    end

    function obj = MultiboxCoder(varargin)
      obj.backgroundLabel = obj.backgroundLabel ;
      obj.load(varargin) ;
    end
  end
end
