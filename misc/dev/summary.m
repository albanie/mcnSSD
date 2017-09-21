function summary(varargin)

  opts.model = 'vggvd' ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  switch opts.model
    case 'vggvd', modelName = 'ssd-pascal-vggvd-300.mat' ;
    case 'vggvd-mcn', modelName = 'ssd-mcn-pascal-vggvd-300.mat' ;
    case 'mobile', modelName = 'ssd-mcn-mobilenet.mat' ;
  end
  modelPath = fullfile(opts.modelDir, modelName) ;

  dag = dagnn.DagNN.loadobj(load(modelPath)) ;

  for ii = 1:numel(dag.layers)
    block = dag.layers(ii).block ;
    type = class(block) ;
    %fprintf('type: %s\n', type) ;
    switch type
      case 'dagnn.Flatten'
        fprintf('flatten first: %d, axis: %d\n', block.firstAxis, block.axis) ;
      case 'dagnn.Permute' 
        fprintf('permute order: [%dx%dx%dx%d]\n', block.order) ;

      case 'dagnn.PriorBox'
        fprintf(['priors: pixelStep %d, minSize: %d, maxSize: %d' ...
                 ' offset: %f\n'], block.pixelStep, block.minSize, block.minSize) ;
      case 'dagnn.Concat'
        %if block.dim == 1, keyboard ; end
        fprintf('concat dim: %d, inputSizes: %g\n', ...
                  block.dim, numel(block.inputSizes)) ;
      case 'dagnn.Reshape'
        %if block.dim == 1, keyboard ; end
        fprintf('reshape shape: ') ;
        disp(block.shape) ;
      case 'dagnn.SoftMaxTranspose'
        fprintf('softmaxT dim: %d\n', block.dim) ;
      case 'dagnn.SoftMax'
        keyboard
    end

  end
