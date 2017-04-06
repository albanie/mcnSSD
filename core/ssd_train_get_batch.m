function batchData = ssd_train_get_batch(imdb, batch, batchOpts, varargin)
% SSD_TRAIN_GET_BATCH generates mini batches for training SSD

imIds = imdb.images.name(batch) ;
imNames = imdb.images.name(batch) ;
imPathTemplates = imdb.images.paths(batch) ;
imPaths = cellfun(@(x,y) sprintf(x, y), imPathTemplates, imNames, 'Uni', 0) ;

annotations = {imdb.annotations{batch}} ;
targets = cellfun(@(x) x.boxes, annotations, 'UniformOutput', false) ;
labels = cellfun(@(x) single(x.classes), annotations, 'UniformOutput', false) ;

% ------------------------------------------
% Data augmentation
% ------------------------------------------

data = single(zeros([batchOpts.imageSize(1:2) 3 numel(batch)])) ;

% add singletons to mean for bsxfun
RGB = [123, 117, 104] ;
imMean = permute(RGB, [3 1 2]) ;

% randomly generate resize methods
resizeMethods = {batchOpts.resizeMethods{randi(numel(batchOpts.resizeMethods), ...
                                               1, numel(batch))}} ;

for i = 1:numel(batch)

    im = single(imread(imPaths{i})) ;
    targets_ = targets{i} ;
    labels_ = labels{i} ;
    sz = size(im) ;


    % apply image distortion
    if batchOpts.distortOpts.use
        if rand < batchOpts.distortOpts.brightnessProb
            delta = batchOpts.distortOpts.brightnessDelta ;
            assert(delta >= 0, 'brightness delta must be non-negative') ;
            adjust = -delta + rand * 2 * delta ;
            % adjust brightness and clip
            im = max(min(im + adjust, 255), 0) ;
        end

        if rand < batchOpts.distortOpts.contrastProb
            lower = batchOpts.distortOpts.contrastLower ;
            upper = batchOpts.distortOpts.contrastUpper ;
            assert(upper >= lower, 'upper contrast must be >= lower') ;
            assert(lower >= 0, 'lower contrast must be non-negative') ;
            adjust = lower + rand * (upper - lower) ;
            % adjust contrast and clip
            im = max(min(im * adjust, 255), 0) ;
        end

        if rand < batchOpts.distortOpts.saturationProb
            lower = batchOpts.distortOpts.saturationLower ;
            upper = batchOpts.distortOpts.saturationUpper ;
            assert(upper >= lower, 'upper saturation must be >= lower') ;
            assert(lower >= 0, 'lower saturation must be non-negative') ;
            adjust = lower + rand * (upper - lower) ;
            im_ = rgb2hsv(im / 255) ;
            % adjust saturation and clip
            sat = max(min(im_(:,:,2) * adjust,  1), 0) ;
            im_(:,:,2) = sat ;
            im = hsv2rgb(im_) * 255 ;
        end

        if rand < batchOpts.distortOpts.hueProb
            delta = batchOpts.distortOpts.hueDelta ;
            assert(delta >= 0, 'hue delta must be non-negative') ;
            adjust = -delta + rand * 2 * delta ;
            im_ = rgb2hsv(im / 255) ;
            % adjust hue and clip
            hue = max(min(im_(:,:,1) + adjust,  1), 0) ;
            im_(:,:,1) = hue ;
            im = hsv2rgb(im_) * 255 ;
        end

        if rand < batchOpts.distortOpts.randomOrderProb
            im = im(:,:,randperm(3)) ;
        end
    end

    % zoom out
    if batchOpts.zoomOpts.use && rand < batchOpts.zoomOpts.prob 
        minScale = batchOpts.zoomOpts.minScale ;
        maxScale = batchOpts.zoomOpts.maxScale ;
        zoomScale = minScale + rand * (maxScale - minScale) ;
        canvasSize = [ round(sz(1:2) * zoomScale) 3 ];
        canvas = bsxfun(@times, ones(canvasSize), permute(RGB, [3 1 2])) ;

        % uniformly sample location from feasible region
        minYX_ = rand(1, 2) ;
        minYX = minYX_ .* (canvasSize(1:2) - sz(1:2)) ;
        yLoc = round(minYX(1) + 1:minYX(1) + sz(1)) ;
        xLoc = round(minYX(2) + 1:minYX(2) + sz(2)) ;

        % insert image at location
        canvas(yLoc,xLoc,:) = im ;

        % update targets
        targetsMinWH = bboxCoder(targets_, 'MinMax', 'MinWH') ;
        updatedWH = targetsMinWH(:,3:4) / zoomScale ;
        offsets = [minYX_(2) minYX_(1)] * (zoomScale - 1 ) / zoomScale ;
        updatedXY = bsxfun(@plus, offsets, targetsMinWH(:,1:2) / zoomScale) ;
        updatedTargetsMinWH = [updatedXY updatedWH ] ;
        targets_ = bboxCoder(updatedTargetsMinWH, 'MinWH', 'MinMax') ;
        im = canvas ;
        sz = size(im) ;
    end

    % sample a patch
    if batchOpts.patchOpts.use
        [patch, targets_, labels_] = patchSampler(targets_, labels_, batchOpts.patchOpts) ;
        % crop to patch 
        xmin = 1 + round(patch(1) * (sz(2) - 1)) ;
        xmax = 1 + round(patch(3) * (sz(2) - 1)) ;
        ymin = 1 + round(patch(2) * (sz(1) - 1)) ;
        ymax = 1 + round(patch(4) * (sz(1) - 1)) ;
        im = im(ymin:ymax, xmin:xmax, :) ;
    end

    % resize to fit network with sampled method
    im = imresize(im, batchOpts.imageSize(1:2), 'method', resizeMethods{i}) ;

    % flipping
    if batchOpts.flipOpts.use && rand < batchOpts.flipOpts.prob
        im = fliplr(im) ;
        targets_ = [1 - targets_(:,3) targets_(:,2) ...
                      1 - targets_(:,1) targets_(:,4) ] ;
    end

    labels{i} = labels_ ;
    targets{i} = targets_ ;
    data(:,:,:,i) = bsxfun(@minus, im, imMean) ;
end

if batchOpts.useGpu
    data = gpuArray(data) ;
end

if 0
    ssd_viz_get_batch(data, labels, targets) ;
end

batchData = {'data', data, ...
             'labels', labels, ...
             'targets', targets } ;
