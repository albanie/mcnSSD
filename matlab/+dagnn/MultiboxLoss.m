classdef MultiboxLoss < dagnn.Loss

  properties
    locWeight = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnmultiboxloss(inputs{1}, inputs{2}, ...
                                    'locWeight', obj.locWeight, ...
                                     obj.opts{:}) ;

      % Accumulate loss statistics across batch
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = vl_nnmultiboxloss(inputs{1}, inputs{2},  derOutputs{1}, ... 
                                    'locWeight', obj.locWeight, obj.opts{:}) ;
      derParams = {} ;
    end

    function obj = LossSmoothL1(varargin)
      obj.load(varargin) ;
      obj.loss = 'smoothl1';
    end
  end
end
