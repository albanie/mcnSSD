function patch = runPatchTrials(targetsWH, strategy, randSource, opts)

% check random samples are appropriate
assert(all(size(randSource) == [opts.numTrials 4]), ...
        'PATCHTRIALS:incorrectRandSize', ...
        'the randSource must contain the required number of samples') ;

assert(max(abs(randSource(:) - 0.5)) <= 0.5, ...
        'PATCHTRIALS:incorrectRandRange', ...
        'the randSource samples must be drawn uniformly from [0,1]') ;

patchScaleRand = randSource(:,1) ;
patchAspectRand = randSource(:,2) ;
offsetRand = randSource(:,3:4) ;

% NOTE: Following the caffe implementation, `scale` here 
% refers to the square root of the sampled patch area.
patchScale = opts.minPatchScale + ...
                (opts.maxPatchScale - opts.minPatchScale) .* patchScaleRand ;

% the aspect ratio is constrained to fit inside the 
% unit box 
minAspect = max(opts.minAspect, patchScale.^2) ;
maxAspect = min(opts.maxAspect, 1 ./ (patchScale.^2)) ;

patchAspect = minAspect + patchAspectRand .* (maxAspect - minAspect);
patchWidth = patchScale .* sqrt(patchAspect) ;
patchHeight = patchScale ./ sqrt(patchAspect) ;

xmax = 1 - patchWidth ;
ymax = 1 - patchHeight ;

% use importance sampling to get a uniform sample from the
% feasible region
XY = bsxfun(@times, offsetRand, [xmax ymax]) ;
potentialBoxes = [ XY patchWidth patchHeight] ;
overlaps = bboxOverlapRatio(targetsWH, potentialBoxes) ;

switch strategy
    case 'rand_patch'
        minOverlap = 0 ;
    case 'jacc_0.1'
        minOverlap = 0.1 ;
    case 'jacc_0.3'
        minOverlap = 0.3 ;
    case 'jacc_0.5'
        minOverlap = 0.5 ;
    case 'jacc_0.7'
        minOverlap = 0.7 ;
    case 'jacc_0.9'
        minOverlap = 0.9 ;
end

%% select patch
matches = find(max(overlaps, [], 1) >= minOverlap) ;

 % if no matches, stick to original sample
if ~length(matches)
    patch = [] ;
else 
    % otherwise return first match
    patchWH = potentialBoxes(matches(1), :) ;
    patch = bboxCoder(patchWH, 'MinWH', 'MinMax') ;
end
