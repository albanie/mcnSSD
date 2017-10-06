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
    if opts.multiscale
      predictions = computePredictionsMultiscale(net, imdb, testIdx, opts) ;
    else
      predictions = computePredictions(net, imdb, testIdx, opts) ;
    end
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
  switch opts.dataOpts.decoder % For small datasets serial tends to be faster
    case 'serial', decodedPreds = decodeSerial(args{:}) ;
    case 'parallel', decodedPreds = decodeParallel(args{:}) ;
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
  numClasses = numel(imdb.meta.classes) -1 ; 

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
        fprintf('extracting preds for %s (%d/%d)\n', ...
                                imdb.meta.classes{target}, c, numClasses) ;
        cBoxes = zeros(numTotal, 4, 'single') ; cImageIds = cell(numTotal, 1) ;
        cScores = zeros(numTotal, 1, 'single') ; keep = false(numTotal, 1) ;

        parfor p = 1:numTotal
          % temp assignments to avoid broadcasting overhead in parfor
          predictions_ = predictions ; imdb_ = imdb ; opts_ = opts ;
          testIdx_ = testIdx ;
          scoreThresh = opts_.dataOpts.scoreThresh ;
          offset = mod(p - 1, predsPerIm) + 1 ; 
          imId = idivide(p - 1, predsPerIm) + 1 ;
          preds = predictions_(offset,:,:,imId) ;
          keep(p) = (preds(1) == target) ; 
          if ~keep(p)
              continue
          else
            pScore = preds(2) ;
            if pScore < scoreThresh, keep(p) = false ; continue ; end

            pBox = preds(3:end) ;
            % clip predictions to fall in image and scale the 
            % bounding boxes from [0,1] to absolute pixel values
            pBox = min(max(pBox, 0), 1) ;
            imsz = single(imdb_.images.imageSizes{testIdx_(imId)}) ;
            minX = pBox(1)*imsz(2) ; minY = pBox(2)*imsz(1) ;
            switch opts.dataOpts.resultsFormat
              case 'minMax'
                maxX = pBox(3)*imsz(2) ; maxY = pBox(4)*imsz(1) ;
                pBox = [ minX minY maxX maxY] ;
              case 'minWH'
                W = (pBox(3) - pBox(1))*imsz(2) ; H = (pBox(4) - pBox(2))*imsz(1)  ;
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
      imsz = arrayfun(@(x) {repmat(imSizes(x,[2 1]),pSize(1),2)},1:numel(testIdx)) ;
      imIds = vertcat(imIds{:}) ; imsz = vertcat(imsz{:}) ;
      pBox = preds(:,end-3:end) ; pBox = min(max(pBox, 0), 1) ;
      switch opts.dataOpts.resultsFormat
          case 'minMax'
            pBox = bsxfun(@times, pBox, imsz) ;
          case 'minWH'
            pBox = bsxfun(@times, [pBox(:,1:2) pBox(:,3:4) - pBox(:,1:2)], imsz) ;
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
function predictions = computePredictionsMultiscale(net, imdb, testIdx, opts) 
% ----------------------------------------------------------------------------
prepareGPUs(opts, true) ; params.testIdx = testIdx ;
predictions = cell(1, numel(opts.msOpts.scales)) ;
scales = opts.msOpts.scales ;
rng(sum(clock)) ; % try to prevent redundant work
scales = scales(randperm(numel(scales))) ; % wild west optimisation
for ii = 1:numel(scales) % need to do this for each scale
  scale = scales(ii) ; 
  % select net - original SSD is stored first, otherwise use modified net
  if scale == 1, net_ = net{1} ; else, net_ = net{2} ; end
  params.predIdx = net_.getVarIndex('detection_out') ;
  % feeling brave
  cacheFile = sprintf('%s-%s-scale-%.2f.mat', ...
                  opts.modelName, opts.testset, scale) ;
  [~,placeHolder,~] = fileparts(cacheFile) ;
  placeHolderFile = sprintf('%s.txt', placeHolder) ;
  cachePath = fullfile(vl_rootnn, 'data/coco17', cacheFile) ;
  placeHolderPath = fullfile(vl_rootnn, 'data/coco17/plc/', placeHolderFile) ;
  if exist(cachePath, 'file')
    tmp = load(cachePath) ;
    predictionsMs = tmp.predictionsMs ;
    fprintf('loaded scale %f from cache \n', scale) ;
  elseif exist(placeHolderPath, 'file')
    fprintf('skipping evaluation of scale: %.2f\n', scale) ;
    continue ;
  else
    if ~exist(fileparts(placeHolderPath), 'dir')
      mkdir(fileparts(placeHolderPath)) ; 
    end
    fid = fopen(placeHolderPath, 'w') ;
    fprintf(fid, 'scale %.2f is being processed \n', scale) ;
    fclose(fid) ;
    fprintf('processing scale: %g\n', scale) ;
    if numel(opts.gpus) <= 1
      addpath(fullfile(vl_rootnn, 'contrib/mcnSSD/matlab/mex')) ; % parallel path issue
       state = processDetections(net_, imdb, params, opts, 'scale', scale) ;
       predictionsMs = state.predictions ;
    else
      predictionsMs = zeros(200, 6, 1, numel(testIdx), 'single') ; 
      addpath(fullfile(vl_rootnn, 'contrib/mcnSSD/matlab/mex')) ; % parallel path issue
      spmd
        state = processDetections(net_, imdb, params, opts, 'scale', scale) ;
      end
      for i = 1:numel(opts.gpus)
        state_ = state{i} ;
        predictionsMs(:,:,:,state_.computedIdx) = state_.predictions ;
      end
    end
    tmp.predictionsMs = predictionsMs ;
    if ~exist(fileparts(cachePath), 'dir'), mkdir(fileparts(cachePath)) ; end
    save(cachePath, '-struct', 'tmp', '-v7.3') ;
  end
  predictions{ii} = predictionsMs ;

  % safe mode
  %state = processDetections(net_, imdb, params, opts, 'scale', scale) ;
  %predictions{ii} = state.predictions ;
end
predictions = mergeMultiscalePredictions(predictions, imdb, opts) ;

% ------------------------------------------------------------------
function predictions = computePredictions(net, imdb, testIdx, opts) 
% ------------------------------------------------------------------
  prepareGPUs(opts, true) ; params.testIdx = testIdx ;
  params.predIdx = net.getVarIndex('detection_out') ;
  if numel(opts.gpus) <= 1
     state = processDetections(net, imdb, params, opts) ;
     predictions = state.predictions ;
  else
    predictions = zeros(200, 6, 1, numel(testIdx), 'single') ; 
    addpath(fullfile(vl_rootnn, 'contrib/mcnSSD/matlab/mex')) ; % parallel path issue
    spmd
      state = processDetections(net, imdb, params, opts) ;
    end
    for i = 1:numel(opts.gpus)
      state_ = state{i} ;
      predictions(:,:,:,state_.computedIdx) = state_.predictions ;
    end
  end

% --------------------------------------------------------------------
function selectedPreds = mergeMultiscalePredictions(preds, imdb, opts) 
% --------------------------------------------------------------------
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

    % deal with offset
    cIdx = ii + 1 ;  
    classPreds_ = [] ;

    % do the merge
    for jj = 1:size(preds_,4)
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
  computedIdx = sort(horzcat(idx{:})) ; keepTopK = net.meta.keepTopK ;
  state.predictions = zeros(keepTopK, 6, 1, numel(computedIdx), 'single') ; 
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
    net.eval(inputs, 'testmemoryless') ;
    storeIdx = offset:offset + numel(batch) - 1 ;offset = offset + numel(batch) ;
    state.predictions(:,:,:,storeIdx) = net.vars{params.predIdx} ;
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
