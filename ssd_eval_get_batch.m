function batchData = ssd_eval_get_batch(imdb, batch, opts, varargin)

bopts.prefetch = false ;
bopts = vl_argparse(bopts, varargin) ;

imMean = [123, 117, 104] ; 
useGpu = numel(opts.gpus) > 0 ;
imNames = imdb.images.name(batch) ;
imPathTemplates = imdb.images.paths(batch) ;
imPaths = cellfun(@(x,y) sprintf(x, y), imPathTemplates, imNames, 'Uni', 0) ;

if opts.batchOpts.use_vl_imreadjpeg
    args = {imPaths, ...
            'Pack', ...
            'Verbose', ...
            'NumThreads', opts.batchOpts.numThreads, ...
            'Interpolation', 'bilinear', ...
            'SubtractAverage', imMean, ...
            'CropAnisotropy', [0 0] ...
            'Resize', opts.batchOpts.imageSize(1:2) ...
            } ;

    if useGpu > 0 
      args{end+1} = {'Gpu'} ;
    end

    args = horzcat(args(1), args{2:end}) ;

    if bopts.prefetch
        vl_imreadjpeg(args{:}, 'prefetch') ;
        data = [] ;
    else
        out = vl_imreadjpeg(args{:}) ;
        data = out{1} ; 
    end
else
    data = single(zeros([opts.batchOpts.imageSize(1:2) 3 numel(batch)])) ;
    imMean = reshape(imMean, 1, 1, 3) ;
    for i = 1:numel(imPaths) 
        im = single(imread(imPaths{i})) ;

        if ~opts.fixedSizeInputs
            im = imresize(im, opts.batchOpts.imageSize(1:2), ...
                          'method', 'bicubic') ;
        end
        data(:,:,:,i) = bsxfun(@minus, im, imMean) ;
    end
    
    if useGpu
        data = gpuArray(data) ;
    end
end
    
batchData = {'data', data} ;
