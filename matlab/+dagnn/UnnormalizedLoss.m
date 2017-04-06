classdef UnnormalizedLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], ...
                                   'loss', obj.loss, ...
                                   'instanceWeights', inputs{3}, ...
                                   obj.opts{:}) ;
      n = obj.numAveraged ;
      m = n + 1 ; % averaging is handled by instance weights
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                                'loss', obj.loss, ...
                                'instanceWeights', inputs{3}, ...
                                obj.opts{:}) ;
      derInputs{2} = [NaN] ;
      derInputs{3} = [NaN] ;
      derParams = {} ;
    end
  end
end
