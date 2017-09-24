classdef PriorBox < dagnn.ElementWise
  properties
    minSize
    maxSize
    aspectRatios
    variance
    pixelStep
    flip
    clip
    offset = 0.5
    usePriorCaching = true
    priorCache = []
  end
  
  methods
    function outputs = forward(obj, inputs, params)
        if obj.usePriorCaching && ~isempty(obj.priorCache)
            outputs = obj.priorCache ;
        else
            y = vl_nnpriorbox(inputs{1}, inputs{2}, ...
                                     'minSize', obj.minSize, ...
                                     'maxSize', obj.maxSize, ...
                                     'aspectRatios', obj.aspectRatios, ...
                                     'variance', obj.variance, ...
                                     'pixelStep', obj.pixelStep, ...
                                     'flip', obj.flip, ...
                                     'clip', obj.clip, ...
                                     'offset', obj.offset) ;

            % move priors to GPU if required
            if isa(inputs{1}, 'gpuArray') ; y = gpuArray(y) ; end
            outputs{1} = y ;
        end  

        % NOTE: if the training is moved from the gpu to the cpu
        % during training, the cache must be refreshed
        if obj.usePriorCaching
            obj.priorCache = outputs ;
        end
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ;
      derParams = {} ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      numAspectRatios = 1 + (1 + obj.flip) * numel(obj.aspectRatios) ...
                                + logical(obj.maxSize) ;
      numBoxes = numAspectRatios * inputSizes{1}(2) * inputSizes{1}(1) ;
      outputSizes{1} = [numBoxes 2 2 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end
    
    function obj = PriorBox(varargin)
      obj.load(varargin{:}) ;
      obj.minSize = obj.minSize ;
      obj.maxSize = obj.maxSize ;
      obj.aspectRatios = obj.aspectRatios ;
      obj.flip = obj.flip ;
      obj.clip = obj.clip ;
      obj.variance = obj.variance ;
    end
  end
end
