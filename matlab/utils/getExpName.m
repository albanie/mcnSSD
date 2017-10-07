function expName = getExpName(modelOpts, dataOpts) 
%GETEXPNAME defines a naming strategy for each experiment, 
% depending on the options used during training

if dataOpts.useValForTraining
    subset = 'vt' ;
else
    subset = 't' ;
end

expName = sprintf('%s-%s-%s-%s-%d-%d', modelOpts.type, ...
                                       dataOpts.name, ...
                                       dataOpts.trainData, ...
                                       subset, ...
                                       modelOpts.batchSize, ...
                                       modelOpts.architecture) ; 

if dataOpts.flipAugmentation
    expName = [ expName '-flip' ] ;
end

if dataOpts.patchAugmentation
    expName = [ expName '-patch' ] ;
end

if dataOpts.zoomAugmentation
    expName = [ expName sprintf('-zoom-%d', dataOpts.zoomScale) ] ;
end

if dataOpts.distortAugmentation
    expName = [ expName '-distort' ] ;
end
