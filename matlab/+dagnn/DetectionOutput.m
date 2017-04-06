classdef DetectionOutput < dagnn.ElementWise
  properties
    numClasses = 21
    shareLocation = true
    nmsThreshold = 0.45
    nmsTopK = 400
    confidenceThreshold = 0.01
    keepTopK = 200
    %codeType = 'CENTER_SIZE'
  end

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = vl_ssdoutput(inputs{1}, inputs{2}, inputs{3}, ...
                                'numClasses', obj.numClasses, ...
                                'nmsThreshold', obj.nmsThreshold, ...
                                'nmsTopK', obj.nmsTopK, ...
                                'confidenceThreshold', obj.confidenceThreshold, ...
                                'keepTopK', obj.keepTopK) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [0,0,0,0] ;
    end

    function rfs = getReceptiveFields(obj)
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Flatten(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
