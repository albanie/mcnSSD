function confirmConfig(expDir, opts)
%DOCS: todo - pretty obvs

if ~opts.confirmConfig
    return ;
end

[~,expName] = fileparts(expDir) ;
fprintf('Experiment name: %s\n', expName) ;
fprintf('------------------------------------\n') ;
fprintf('Training set: %s\n', opts.dataOpts.trainData) ;
fprintf('Testing set: %s\n', opts.dataOpts.testData) ;
fprintf('------------------------------------\n') ;
fprintf('Prune checkpoints: %d\n', opts.pruneCheckpoints) ;
fprintf('GPU: %d\n', opts.train.gpus) ;
fprintf('Batch size: %d\n', opts.train.batchSize) ;
fprintf('------------------------------------\n') ;
fprintf('Train + val: %d\n', opts.dataOpts.useValForTraining) ;
fprintf('Flip: %d\n', opts.dataOpts.flipAugmentation) ;
fprintf('Patches: %d\n', opts.dataOpts.patchAugmentation) ;
fprintf('Zoom: %d\n', opts.dataOpts.zoomAugmentation) ;
fprintf('Distort: %d\n', opts.dataOpts.distortAugmentation) ;
fprintf('------------------------------------\n') ;
printSchedule(opts) ;
fprintf('------------------------------------\n') ;

waiting = true ;
prompt = 'Run experiment with these parameters? `y` or `n`\n' ;

while waiting
    str = input(prompt,'s') ;
    switch str
        case 'y'
            return ;
        case 'n'
            throw(exception) ;
        otherwise
            fprintf('input %s not recognised, please use `y` or `n`\n', str) ;
    end
end

% -------------------------
function printSchedule(opts) 
% -------------------------
% format learning rates (assumes monotonically 
% decreasing  after the first warmup epochs)

fprintf('Learning Rate Schedule: \n') ;

lr = opts.train.learningRate ;
warmup = lr(1:2) ;
fprintf('%g %g (warmup)\n', warmup(1), warmup(2)) ;
schedule = lr(3:end) ;
[tiers, uIdx] = unique(schedule) ;
[~, order] = sort(uIdx) ;
for i = 1:numel(order) 
    idx = uIdx(order)' ;
    tierLength = [ idx(2:end) numel(schedule) + 1] - idx ;
    fprintf('%g for %d epochs\n', tiers(order(i)), tierLength(i)) ;
end

