function results = ssd_evaluation(expDir, net, opts)
%SSD_EVALUATION - run SSD detector evaluation
%  SSD_EVALUATION(EXPDIR, NET, OPTS) - evaluates the network NET
%  on the imdb specified (as an option), and stores results in 
%  EXPDIR.
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  % load/create imdb and configure
  if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
  else
    imdb = opts.dataOpts.getImdb(opts) ;
    imdbDir = fileparts(opts.dataOpts.imdbPath) ;
    if ~exist(imdbDir, 'dir'), mkdir(imdbDir) ; end
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
  end
  [opts, imdb] = opts.dataOpts.configureImdbOpts(expDir, opts, imdb) ;

  switch opts.testset
    case 'train', setLabel = 1 ;
    case 'val', setLabel = 2 ;
    case 'test', setLabel = 3 ;
    case 'test-dev', setLabel = 4 ;
  end
  testIdx = find(imdb.images.set == setLabel) ;
  % retrieve results from cache if possible
  results = checkCache(opts, net, imdb, testIdx) ;
  opts.dataOpts.displayResults(opts.modelName, results, opts) ;

% -------------------------------------------------
function res = checkCache(opts, net, imdb, testIdx)
% -------------------------------------------------
  path = opts.cacheOpts.resultsCache ;
  if exist(path, 'file') && ~opts.cacheOpts.refreshCache
    fprintf('loading results from cache\n') ;
    tmp = load(path) ; res = tmp.results ;
  else
    predictions = computePredictions(net, imdb, testIdx, opts) ;
    decodedPreds = decodePredictions(predictions, imdb, testIdx, opts) ;
    s.results = opts.dataOpts.eval_func(opts.modelName, decodedPreds, imdb, opts) ;
    fprintf('saving results to %s\n', path);
    save(path, '-struct', 's', '-v7.3') ;
    res = s.results ;
  end

% -------------------------------------------------------------------------
function decodedPreds = decodePredictions(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
  args = {predictions, imdb, testIdx, opts} ;
  switch opts.dataOpts.decoder 
    % For debuggin/small datasets serial decoding is useful
    case 'serial', decodedPreds = decodeSerial(args{:}) ;
    case 'parallel', decodedPreds = decodeParallel(args{:}) ;
    case 'custom', decodedPreds = opts.dataOpts.customDecoder(args{:}) ;
    otherwise, error('deocoder %s not recognised',opts.dataOpts.deocoder) ;
  end

% -------------------------------------------------------------------------
function decodedPreds = decodeSerial(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
  numClasses = numel(imdb.meta.classes) ; 
  % gather predictions by class, and store the corresponding 
  % image id (i.e. image name) and bouding boxes
  imageIds = cell(1, numClasses) ;
  scores = cell(1, numClasses) ;
  bboxes = cell(1, numClasses) ;
  for c = 1:numClasses
    fprintf('extracting predictions for %s\n', imdb.meta.classes{c}) ;
    for p = 1:numel(testIdx)
      target = c + 1 ; % add offset for bg class (compatibility with caffe)
      % find predictions for current image
      preds = predictions(:,:,:,p) ; targetIdx = find(preds(:,1) == target) ; 
      pScores = preds(targetIdx, 2) ; pBoxes = preds(targetIdx, 3:end) ;

      % clip predictions to fall in image and scale the 
      % bounding boxes from [0,1] to absolute pixel values
      if pBoxes
        pBoxes = min(max(pBoxes, 0), 1) ;
        imsz = single(imdb.images.imageSizes{testIdx(p)}) ;
        switch opts.dataOpts.resultsFormat
            case 'minMax'
                pBoxes = [pBoxes(:,1) * imsz(2) ...
                          pBoxes(:,2) * imsz(1) ...
                          pBoxes(:,3) * imsz(2) ...
                          pBoxes(:,4) * imsz(1) ] ;
            case 'minWH'
                pBoxes = [pBoxes(:,1) * imsz(2) ...
                          pBoxes(:,2) * imsz(1) ...
                          (pBoxes(:,3) - pBoxes(:,1)) * imsz(2) ...
                          (pBoxes(:,4) - pBoxes(:,2)) * imsz(1) ] ;
        end
        pId = imdb.images.name{testIdx(p)} ; % store results
        scores{c} = vertcat(scores{c}, pScores) ;
        bboxes{c} = vertcat(bboxes{c}, pBoxes) ;
        imageIds{c} = vertcat(imageIds{c}, repmat({pId}, size(pScores))) ; 
      end
      if mod(p,100) == 1 
        fprintf('extracting preds for image %d/%d\n', p, numel(testIdx)) ;
      end
    end
  end
  decodedPreds.imageIds = imageIds ; 
  decodedPreds.scores = scores ;
  decodedPreds.bboxes = bboxes ;

% -------------------------------------------------------------------------
function decodedPreds = decodeParallel(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
  numClasses = numel(imdb.meta.classes) - 1 ; 
  % gather predictions by class, and store the corresponding 
  % image id (i.e. image name) and bouding boxes
  imageIds = cell(1, numClasses) ;
  scores = cell(1, numClasses) ;
  bboxes = cell(1, numClasses) ;

  switch opts.dataOpts.name
    case 'pascal'
      % Use preallocation to enable parallel evaluation
      predsPerIm = uint32(size(predictions, 1)) ;
      numTotal = double(predsPerIm) * numel(testIdx) ;
      for c = 1:numClasses
        target = c + 1 ; % add offset for bg class (compatibility with caffe)
        cBoxes = zeros(numTotal, 4, 'single') ; cImageIds = cell(numTotal, 1) ;
        cScores = zeros(numTotal, 1, 'single') ; keep = false(numTotal, 1) ;
        fprintf('extracting predictions for class (%d/%d) \n', c, numClasses) ;

        parfor p = 1:numTotal
          % temp assignments to avoid broadcasting overhead in parfor
          predictions_ = predictions ; imdb_ = imdb ; opts_ = opts ;
          testIdx_ = testIdx ;
          offset = mod(p - 1, predsPerIm) + 1 ; 
          imId = idivide(p - 1, predsPerIm) + 1 ;
          preds = predictions_(offset,:,:,imId) ;
          keep(p) = (preds(1) == target) ; 
          if ~keep(p)
              continue
          else
            pScore = preds(2) ; pBox = preds(3:end) ;
            % clip predictions to fall in image and scale the 
            % bounding boxes from [0,1] to absolute pixel values
            pBox = min(max(pBox, 0), 1) ;
            imsz = single(imdb_.images.imageSizes{testIdx_(imId)}) ;
            minX = pBox(1)*imsz(2) ; minY = pBox(2)*imsz(1) ;
            switch opts_.dataOpts.resultsFormat
              case 'minMax'
                maxX = pBox(3)*imsz(2) ; maxY = pBox(4)*imsz(1) ;
                pBox = [ minX minY maxX maxY] ;
              case 'minWH'
                W = (pBox(3) - pBox(1))*imsz(2) ; 
                H = (pBox(4) - pBox(2))*imsz(1)  ;
                pBox = [ minX minY W H] ;
            end
            % store results
            cScores(p) = pScore ; cBoxes(p,:) = pBox ;
            cImageIds{p} = imdb_.images.name{testIdx_(imId)} ; 
          end
        end
        scores{c} = cScores(keep) ;
        bboxes{c} = cBoxes(keep,:) ;
        imageIds{c} = cImageIds(keep) ; 
      end
    case 'coco'
      % to enable parallel processing on coco, the predictions are flattened
      pSize = size(predictions) ;
      imIds = arrayfun(@(x) {repmat(x, pSize(1), 1)}, imdb.images.id(testIdx)) ;
      preds = reshape(permute(predictions, [1 4 3 2]), [], pSize(2), 1) ;
      imSizes = vertcat(imdb.images.imageSizes{testIdx}) ; 
      range = 1:numel(testIdx) ;
      imsz = arrayfun(@(x) {repmat(imSizes(x,[2 1]), pSize(1),2)}, range) ;
      imIds = vertcat(imIds{:}) ; imsz = vertcat(imsz{:}) ;
      pBox = preds(:,end-3:end) ; pBox = min(max(pBox, 0), 1) ;
      switch opts.dataOpts.resultsFormat
          case 'minMax'
            pBox = bsxfun(@times, pBox, imsz) ;
          case 'minWH'
            canonical = [pBox(:,1:2) pBox(:,3:4) - pBox(:,1:2)] ;
            pBox = bsxfun(@times, canonical, imsz) ;
      end
      pBox = round(pBox, 2) ; % reduce storage requirements
      preds(:,end-3:end) = pBox ; preds = [imIds preds] ;
      parfor c = 1:numClasses
        imdb_ = imdb ; preds_ = preds ; % avoid parfor broadcast indexing
        target = c + 1 ; % add offset for background class
        template = 'extracting predictions for %s (%d/%d)\n' ;
        fprintf(template, imdb_.meta.classes{target}, c, numClasses) ;
        keep = find(preds_(:,2) == target) ;
        imageIds{c} = preds_(keep,1) ; scores{c} = preds_(keep,end-4) ;
        bboxes{c} = preds_(keep,end-3:end) ;
      end
    otherwise, error('dataset %s not recognised\n', opts.dataOpts.name) ;
  end
  decodedPreds.imageIds = imageIds ;
  decodedPreds.scores = scores ;
  decodedPreds.bboxes = bboxes ;

% ----------------------------------------------------------------------------
function predictions = computePredictions(net, imdb, testIdx, opts) 
% ----------------------------------------------------------------------------
  prepareGPUs(opts, true) ; params.testIdx = testIdx ;
  scales = opts.msOpts.scales ; if ~opts.msOpts.use, scales = 1 ; end
  predictions = cell(1, numel(scales)) ;
  for ii = 1:numel(scales) % need to do this for each scale
    scale = scales(ii) ; 
    % select net - original SSD is stored first, otherwise use modified net
    if scale == 1, net_ = net{1} ; else, net_ = net{2} ; end
    params.predIdx = net_.getVarIndex(opts.modelOpts.predVar) ;
    addpath(fullfile(vl_rootnn, 'contrib/mcnSSD/matlab/mex')) ; % fix path issue
    if numel(opts.gpus) <= 1
       state = processDetections(net_, imdb, params, opts, 'scale', scale) ;
       predictions_ = state.predictions ;
    else
      keepTopK = opts.modelOpts.keepTopK ; outCols = opts.modelOpts.outCols ;
      predictions_ = zeros(keepTopK, outCols, 1, numel(testIdx), 'single') ; 
      spmd
        state = processDetections(net_, imdb, params, opts, 'scale', scale) ;
      end
      for i = 1:numel(opts.gpus)
        state_ = state{i} ;
        predictions_(:,:,:,state_.computedIdx) = state_.predictions ;
      end
    end
    predictions{ii} = predictions_ ;
  end
  predictions = mergeMultiscalePredictions(predictions, imdb, opts) ;

% --------------------------------------------------------------------
function selectedPreds = mergeMultiscalePredictions(preds, imdb, opts) 
% --------------------------------------------------------------------
  % handle single scale evaluation first 
  if numel(preds) == 1, selectedPreds = preds{1} ; return ; end

  selectedPreds = zeros(size(preds{1}), 'like', preds{1}) ;
  batchSize = size(selectedPreds, 4) ;
  numClasses = numel(imdb.meta.classes) - 1 ;
  mergeFactor = numel(preds) ; 
  allPreds = cat(4, preds{:}) ; % merge all preds

  % loop over batch and combine
  for bb = 1:batchSize
    classPreds = cell(1, numClasses) ;

    % try not to screw up indexing
    keep = bb:batchSize:mergeFactor*batchSize ;
    preds_ = cat(4, allPreds(:,:,:,keep)) ;
    fprintf('%d/%d\n', bb, batchSize) ;

    % run NMS on the collection (per class)
    for ii = 1:numClasses
      cIdx = ii + 1 ;  classPreds_ = [] ; % deal with offset
      for jj = 1:size(preds_,4) % do the merge
        % select all predictions for class
        classKeep = find(preds_(:,1,:,jj) == cIdx) ;
        classPreds_ = [ classPreds_ ; preds_(classKeep,:,:,jj) ] ;
      end

      % rearrange for nms (scores in the last column)
      if ~isempty(classPreds_)
        candidates = classPreds_(:, [3 4 5 6 2]) ; 
        keepIdx = vl_nms(candidates, ...
                         'overlapThreshold', opts.msOpts.nmsThresh, ...
                         'topK', size(preds_,1)) ;
        classPreds_ = classPreds_(keepIdx,:) ;
       end
      classPreds{ii} = classPreds_ ;
    end

    % recombine and pick top elements by confidence
    combined = vertcat(classPreds{:}) ;
    sortedCombined = sortrows(combined, -2) ;
    numKeep = min(size(preds_,1), size(sortedCombined, 1)) ;
    selectedPreds(1:numKeep,:,:,bb) = sortedCombined(1:numKeep,:) ;
  end

% -------------------------------------------------------------------
function state = processDetections(net, imdb, params, opts, varargin) 
% -------------------------------------------------------------------
  sopts.scale = [] ;
  sopts = vl_argparse(sopts, varargin) ;

  % benchmark speed
  num = 0 ; adjustTime = 0 ; stats.time = 0 ; stats.num = num ;  start = tic ;
  testIdx = params.testIdx ;
  if ~isempty(opts.gpus), net.move('gpu') ; end

  % pre-compute the indices of the predictions made by each worker
  startIdx = labindex:numlabs:opts.batchOpts.batchSize ;
  idx = arrayfun(@(x) {x:opts.batchOpts.batchSize:numel(testIdx)}, startIdx) ;
  computedIdx = sort(horzcat(idx{:})) ; keepTopK = opts.modelOpts.keepTopK ;
  outCols = opts.modelOpts.outCols ;
  state.predictions = zeros(keepTopK, outCols, 1, numel(computedIdx), 'single') ; 
  state.computedIdx = computedIdx ; offset = 1 ;

  for t = 1:opts.batchOpts.batchSize:numel(testIdx) 
    % display progress
    progress = fix((t-1) / opts.batchOpts.batchSize) + 1 ;
    totalBatches = ceil(numel(testIdx) / opts.batchOpts.batchSize) ;
    fprintf('evaluating batch %d / %d: ', progress, totalBatches) ;
    batchSize = min(opts.batchOpts.batchSize, numel(testIdx) - t + 1) ;
    batchStart = t + (labindex - 1) ;
    batchEnd = min(t + opts.batchOpts.batchSize - 1, numel(testIdx)) ;
    batch = testIdx(batchStart : numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    args = {imdb, batch, opts} ;
    if ~isempty(sopts.scale), args = [args, {sopts.scale}] ; end %#ok
    inputs = opts.modelOpts.get_eval_batch(args{:}) ;

    if opts.batchOpts.prefetch
      batchStart_ = t + (labindex - 1) + opts.batchOpts.batchSize ;
      batchEnd_ = min(t + 2*opts.batchOpts.batchSize - 1, numel(testIdx)) ;
      nextBatch = testIdx(batchStart_: numlabs : batchEnd_) ;
      args = {imdb, nextBatch, opts} ;
      if ~isempty(sopts.scale), args = [args, {sopts.scale}] ; end %#ok
      opts.modelOpts.get_eval_batch(args{:}, 'prefetch', true) ;
    end
    net.eval(inputs, 'test') ; % TODO(samuel) return to memoryless when available
    storeIdx = offset:offset + numel(batch) - 1 ;
    offset = offset + numel(batch) ; out = net.vars{params.predIdx} ;
    state.predictions(:,:,:,storeIdx) = out(1:opts.modelOpts.keepTopK,:,:,:) ;
    time = toc(start) + adjustTime ; batchTime = time - stats.time ;
    stats.num = num ; stats.time = time ; currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*opts.batchOpts.batchSize + 1
      % compensate for the first three iterations, which are outliers
      adjustTime = 4*batchTime - time ; stats.time = time + adjustTime ;
    end
    fprintf('speed %.1f (%.1f) Hz', averageSpeed, currentSpeed) ; fprintf('\n') ;
  end
  net.move('cpu') ;

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
  clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
  numGpus = numel(opts.gpus) ;
  if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
      delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
      parpool('local', numGpus) ;
      cold = true ;
    end
  end
  if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    clearMex() ;
    if numGpus == 1
      gpuDevice(opts.gpus)
    else
      spmd
        clearMex() ;
        gpuDevice(opts.gpus(labindex))
      end
    end
  end
