function [patch, targets, labels] = patchSampler(targets, labels, opts) 
% TODO: docs

strategies = {'jacc_0.1', ...
              'jacc_0.3', ...
              'jacc_0.5', ...
              'jacc_0.7', ...
              'jacc_0.9', ...
              'rand_patch'} ;

targetsWH = bboxCoder(targets, 'MinMax', 'MinWH') ;

success = false ;
while ~success
    sampledPatches = cellfun(@(x) runSampleStrategy(x, targetsWH, opts), ...
                            strategies, 'Uni', 0) ;

    % add original image as the last patch strategy
    sampledPatches{end + 1} = [ 0 0 1 1 ] ;

    % remove failed samples
    empty = cellfun(@isempty, sampledPatches) ;
    sampledPatches(empty) = [] ;

    % uniformly sample from patch samples :)
    patch = sampledPatches{randi(length(sampledPatches), 1)} ;

    %----------------------------
    % DEBUG
    cond = patch(:,3:4) - patch(:,1:2) > 0 ;
    assert(all(cond(:)), ...
            'PATCHSAMPLER:invalidPatch', ...
            'patches must be in the (xmin, ymin, xmax, ymax) format') ;
    %----------------------------

    [success, targets, labels] = updateAnnotations(patch, targetsWH, labels, opts) ;
end

cond = targets(:,3:4) - targets(:,1:2) > 0 ;
assert(all(cond(:)), ...
        'PATCHSAMPLER:invalidTargets', ...
        'target boxes must be in the (xmin, ymin, xmax, ymax) format') ;


% ---------------------------------------------------------
function patch = runSampleStrategy(strategy, targetsWH, opts)
% ---------------------------------------------------------

randSource = rand(opts.numTrials, 4) ;
patch = runPatchTrials(targetsWH, strategy, randSource, opts) ;
