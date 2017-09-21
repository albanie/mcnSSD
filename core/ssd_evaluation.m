function ssd_evaluation(expDir, net, opts)

% ----------------------------------------------------------------
%                                                     Prepare imdb
% ----------------------------------------------------------------

if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
else
    imdb = opts.dataOpts.getImdb(opts) ;
    imdbDir = fileparts(opts.dataOpts.imdbPath) ;
    if ~exist(imdbDir, 'dir') 
        mkdir(imdbDir) ;
    end
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
end

opts = opts.dataOpts.configureImdbOpts(expDir, opts) ;

switch opts.testset
    case 'val'
        setLabel = 2 ;
    case 'test'
        setLabel = 3 ;
end

testIdx = find(imdb.images.set == setLabel) ;

% ----------------------------------------------------------------
%                                 Retrieve from caches if possible
% ----------------------------------------------------------------

rawPreds = checkCache('rawPreds', opts, net, imdb, testIdx) ;
decoded = checkCache('decodedPreds', opts, rawPreds, imdb, testIdx) ;
results = checkCache('results', opts, opts.modelName, decoded, imdb) ;

opts.dataOpts.displayResults(opts.modelName, results, opts) ;

% ------------------------------------------------
function res = checkCache(varname, opts, varargin)
% ------------------------------------------------

switch varname
    case 'rawPreds'
        path = opts.cacheOpts.rawPredsCache ;
        flag = opts.cacheOpts.refreshPredictionCache ;
        func = @computePredictions ;
    case 'decodedPreds'
        path = opts.cacheOpts.decodedPredsCache ;
        flag =  opts.cacheOpts.refreshDecodedPredCache ;
        func = @decodePredictions ;
    case 'results'
        path = opts.cacheOpts.resultsCache ;
        flag =  opts.cacheOpts.refreshEvaluationCache ;
        func = opts.dataOpts.eval_func ;
end

if exist(path, 'file') && ~flag
    fprintf('loading %s from cache\n', varname) ;
    tmp = load(path) ;
    res = tmp.(varname) ;
else
    s.(varname) = func(varargin{:}, opts) ;
    save(path, '-struct', 's', '-v7.3') ;
    res = s.(varname) ;
end

% -------------------------------------------------------------------------
function decodedPreds = decodePredictions(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------
% For small datasets (e.g. just a few thousand frames), serial 
% decoding tends to be faster

switch opts.dataOpts.decoder
    case 'serial' 
        decodedPreds = decodeSerial(predictions, imdb, testIdx, opts) ;
    case 'parallel'
        decodedPreds = decodeParallel(predictions, imdb, testIdx, opts) ;
    otherwise
        error('deocoder %s not recognised (should be serial or parallel)', ...
                                                    opts.dataOpts.deocoder) ;
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

        % add offset for background class (for compatibility with caffe)
        target = c + 1 ;

        % find predictions for current image
        preds = predictions(:,:,:,p) ;
        targetIdx = find(preds(:,1) == target) ; 
        pScores = preds(targetIdx, 2) ;
        pBoxes = preds(targetIdx, 3:end) ;

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

            % store results
            pId = imdb.images.name{testIdx(p)} ;
            scores{c} = vertcat(scores{c}, pScores) ;
            bboxes{c} = vertcat(bboxes{c}, pBoxes) ;
            imageIds{c} = vertcat(imageIds{c}, repmat({pId}, size(pScores))) ; 
        end
        fprintf('extracting predictions for image %d/%d\n', p, numel(testIdx)) ;
    end
end

decodedPreds.imageIds = imageIds ;
decodedPreds.scores = scores ;
decodedPreds.bboxes = bboxes ;

% -------------------------------------------------------------------------
function decodedPreds = decodeParallel(predictions, imdb, testIdx, opts) 
% -------------------------------------------------------------------------

numClasses = numel(imdb.meta.classes) ; 
% gather predictions by class, and store the corresponding 
% image id (i.e. image name) and bouding boxes
imageIds = cell(1, numClasses) ;
scores = cell(1, numClasses) ;
bboxes = cell(1, numClasses) ;

% Use preallocation to enable parallel evaluation
predsPerIm = uint32(size(predictions, 1)) ;
numTotal = double(predsPerIm) * numel(testIdx) ;

for c = 1:numClasses
    fprintf('extracting predictions for %s\n', opts.dataOpts.classes{c}) ;

    % add offset for background class (for compatibility with caffe)
    target = c + 1 ;

    cBoxes = zeros(numTotal, 4, 'single') ;
    cScores = zeros(numTotal, 1, 'single') ;
    cImageIds = cell(numTotal, 1) ;
    keep = false(numTotal, 1) ;

    parfor p = 1:numTotal

        offset = mod(p - 1, predsPerIm) + 1 ; 
        imId = idivide(p - 1, predsPerIm) + 1 ;

        preds = predictions(offset,:,:,imId) ;
        keep(p) = (preds(1) == target) ; 
        if ~keep(p)
            continue
        else
            pScore = preds(2) ;
            if pScore < opts.dataOpts.scoreThresh
                keep(p) = false ;
                continue 
            end

            pBox = preds(3:end) ;
            % clip predictions to fall in image and scale the 
            % bounding boxes from [0,1] to absolute pixel values
            pBox = min(max(pBox, 0), 1) ;
            imsz = single(imdb.images.imageSizes{testIdx(p)}) ;

            switch opts.dataOpts.resultsFormat
                case 'minMax'
                    pBox = [pBox(1) * imsz(2) ...
                            pBox(2) * imsz(1) ...
                            pBox(3) * imsz(2) ...
                            pBox(4) * imsz(1) ] ;
                case 'minWH'
                    pBox = [pBox(1) * imsz(2) ...
                            pBox(2) * imsz(1) ...
                            (pBox(3) - pBox(1)) * imsz(2) ...
                            (pBox(4) - pBox(2)) * imsz(1) ] ;
            end

            % store results
            cScores(p) = pScore ;
            cBoxes(p,:) = pBox ;
            cImageIds{p} = imdb.images.name{testIdx(imId)} ; 
        end
        fprintf('extracting prediction %d/%d for %s\n', p, numTotal, ...
                                                  opts.dataOpts.classes{c}) ;
    end
    scores{c} = cScores(find(keep)) ;
    bboxes{c} = cBoxes(find(keep),:) ;
    imageIds{c} = cImageIds(find(keep)) ; 
end

decodedPreds.imageIds = imageIds ;
decodedPreds.scores = scores ;
decodedPreds.bboxes = bboxes ;

% ------------------------------------------------------------------
function predictions = computePredictions(net, imdb, testIdx, opts) 
% ------------------------------------------------------------------

prepareGPUs(opts, true) ;

if numel(opts.gpus) <= 1
   state = processDetections(net, imdb, testIdx, opts) ;
   predictions = state.predictions ;
else
    predictions = zeros(200, 6, 1, numel(testIdx), 'single') ; 
    spmd
       % resolve parallel path issue
       addpath(fullfile(vl_rootnn, 'contrib/mcnSSD/matlab/mex')) ; 
       state = processDetections(net, imdb, testIdx, opts) ;
    end
    for i = 1:numel(opts.gpus)
        state_ = state{i} ;
        predictions(:,:,:,state_.computedIdx) = state_.predictions ;
    end
end

% ------------------------------------------------------------------
function state = processDetections(net, imdb, testIdx, opts) 
% ------------------------------------------------------------------

% benchmark speed
num = 0 ;
adjustTime = 0 ;
stats.time = 0 ;
stats.num = num ; 
start = tic ;

% find the output predictions made by the network
outVars = net.getOutputs() ;
predVar = outVars{1} ;

% prepare to store network predictions
predIdx = net.getVarIndex(opts.modelOpts.predVar) ;
net.vars(predIdx).precious = true ;

if ~isempty(opts.gpus), net.move('gpu') ; end

% pre-compute the indices of the predictions made by each worker
startIdx = labindex:numlabs:opts.batchOpts.batchSize ;
idx = arrayfun(@(x) {x:opts.batchOpts.batchSize:numel(testIdx)}, startIdx) ;
computedIdx = sort(horzcat(idx{:})) ;

% top 200 preds kept
keepTopK = net.layers(net.getLayerIndex('detection_out')).block.keepTopK ;
state.predictions = zeros(keepTopK, 6, 1, numel(computedIdx), 'single') ; 
state.computedIdx = computedIdx ;
offset = 1 ;

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

    inputs = opts.modelOpts.get_eval_batch(imdb, batch, opts) ;

    if opts.prefetch
        batchStart_ = t + (labindex - 1) + opts.batchOpts.batchSize ;
        batchEnd_ = min(t + 2*opts.batchOpts.batchSize - 1, numel(testIdx)) ;
        nextBatch = testIdx(batchStart_: numlabs : batchEnd_) ;
        opts.modelOpts.get_eval_batch(imdb, nextBatch, opts, 'prefetch', true) ;
    end

    net.eval(inputs) ;
    storeIdx = offset:offset + numel(batch) -1 ;
    offset = offset + numel(batch) ;
    state.predictions(:,:,:,storeIdx) = net.vars(predIdx).value ;

    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;

    if t == 3*opts.batchOpts.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    fprintf('speed %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    fprintf('\n') ;
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
