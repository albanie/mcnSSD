function pascal_ssd_experiments(expIdx, gpuIdx)
%PASCAL_EXPERIMENTS

  train.gpus = gpuIdx ;
  train.continue = 0 ;
  modelOpts.architecture = 300 ;

  switch expIdx
    case 1
      dataOpts.trainData = '07' ;
      dataOpts.zoomAugmentation = 0 ; 
      dataOpts.useValForTraining = false ;
    case 2
      dataOpts.trainData = '07' ;
      dataOpts.zoomAugmentation = 1 ; 
      dataOpts.useValForTraining = false ;
    case 3 
      dataOpts.trainData = '0712' ;
      dataOpts.zoomAugmentation = 0 ; 
      dataOpts.useValForTraining = false ;
    case 4 
      dataOpts.trainData = '0712' ;
      dataOpts.zoomAugmentation = 0 ; 
      dataOpts.useValForTraining = true ;
  end

  ssd_pascal_train('train', train, 'dataOpts', dataOpts, ...
                   'modelOpts', modelOpts, 'confirmConfig', 0) ;
