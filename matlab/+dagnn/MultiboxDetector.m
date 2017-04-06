classdef MultiboxDetector < dagnn.ElementWise
  properties
    keepTopK = 200
    nmsTopK = 400
    nmsThresh = 0.45
    confThresh = 0.01
    numClasses = 21
    backgroundLabel = 1 
  end

  methods
    function outputs = forward(obj, inputs, params)
       outputs{1} = vl_nnmultiboxdetector(inputs{1}, inputs{2}, inputs{3}, ...
                              'nmsTopK', double(obj.nmsTopK), ...
                              'keepTopK', double(obj.keepTopK), ...
                              'nmsThresh', obj.nmsThresh, ...
                              'numClasses', double(obj.numClasses), ...
                              'confThresh', obj.confThresh, ...
                              'backgroundLabel', double(obj.backgroundLabel)) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      error('MultiBox detector is only for use at test time') ;
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

    function obj = MultiboxDetector(varargin)
      obj.load(varargin{:}) ;
      obj.backgroundLabel = obj.backgroundLabel ;
    end
  end
end
