function bestEpoch = findBestCheckpoint(expDir, priorityMetric)

lastEpoch = findLastCheckpoint(expDir) ;

% handle the different storage structures/error metrics
data = load(fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)));
if isfield(data, 'stats')
    valStats = data.stats.val;
elseif isfield(data, 'info')
    valStats = data.info.val;
else
    error('storage structure not recognised');
end

% find best checkpoint according to the following priority
metrics = {priorityMetric, 'top1error', 'error', 'mbox_loss', 'class_loss'} ;

for i = 1:numel(metrics)
    if isfield(valStats, metrics{i})
        errorMetric = [valStats.(metrics{i})] ;
        break ;
    end
end

assert(logical(exist('errorMetric')), 'error metrics not recognized') ;
[~, bestEpoch] = min(errorMetric);
