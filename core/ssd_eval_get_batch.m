function out = ssd_eval_get_batch(imdb, batch, opts, scale, varargin)
%SSD_EVAL_GET_BATCH load image batch for inference
%  SSD_EVAL_GET_BATCH(IMDB, BATCH, OPTS, SCALE) loads and preprocess images
%  with the ids specified in the array BATCH from the image database IMDB.
%  The images are resized by the scaling factor SCALE.
%
%   SSD_EVAL_GET_BATCH(..'name', value) accepts the following options:
%
%   `prefetch` :: 'false'
%    If true, "prefetch" the images with vl_imreadjpeg, rather than loading
%    them directly to memory.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  bopts.prefetch = false ;
  bopts = vl_argparse(bopts, varargin) ;

  imMean = opts.batchOpts.imMean ; useGpu = numel(opts.gpus) > 0 ;
  imNames = imdb.images.name(batch) ; paths = imdb.images.paths(batch) ;
  imPaths = cellfun(@(x,y) sprintf(x, y), paths, imNames, 'Uni', 0) ;
  imSize = opts.batchOpts.imageSize(1:2) * scale ;
  args = {imPaths, 'Pack', 'Verbose', ...
          'NumThreads', opts.batchOpts.numThreads, ...
          'Interpolation', 'bilinear', ...
          'SubtractAverage', imMean, ...
          'CropAnisotropy', [0 0] ...
          'Resize', imSize} ;
  if useGpu > 0 , args{end+1} = {'Gpu'} ; end
  args = horzcat(args(1), args{2:end}) ;
  if bopts.prefetch
    vl_imreadjpeg(args{:}, 'prefetch') ; data = [] ;
  else
    out = vl_imreadjpeg(args{:}) ; data = out{1} ; 
  end
  if opts.batchOpts.scaleInputs, data = data * opts.batchOpts.scaleInputs ; end
  out = {'data', data} ;
