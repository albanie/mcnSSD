function pascal_experiments(expIdx, gpuIdx)
%PASCAL_EXPERIMENTS

  % minimal training setup on 2007
  train.gpus = gpuIdx ;
  train.continue = 0 ;
  dataOpts.useValForTraining = false ;
  dataOpts.trainData = '07' ;
  modelOpts.architecture = 300 ;

  switch expIdx
    case 1, dataOpts.zoomAugmentation = 0 ; 
    case 2, dataOpts.zoomAugmentation = 1 ; 
  end

  ssd_pascal_train('train', train, 'dataOpts', dataOpts, ...
                   'modelOpts', modelOpts, 'confirmConfig', 0) ;
