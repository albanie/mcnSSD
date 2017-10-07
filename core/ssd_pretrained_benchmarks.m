% This script evaluates the pre-trained models released by the 
% mastermind behind the SSD detector, Wei Liu, as well as some models
% trained with the matconvnet implementation and the mobilenet detector 
% released by chuanqi305 on GitHub.  It also evaluates some models on 
% MS COCO.
%
% Some notes about the pretrained models/Pascal VOC evaluation
% -------------------------------------------------------------
%
% The pascal evaluation is done on the VOC 2007 test data. 
%
% There are two official evaluation scripts. Pre-2010, the average 
% precision was computed at a set of fixed interval points.  From 2010, 
% the computation was changed to perform numerical integration across 
% the whole curve (details of the change can be found here (Section 3.4.1): 
% http://host.robots.ox.ac.uk/pascal/VOC/voc2010/htmldoc/devkit_doc.html

% All evaluations on the 2007 data should use the 2007 11 point mAP metric.
% However, the `official` evaluation code is extremely slow. There is an 
% unofficial `fast` script which is useful for development (and which should
% produce identical results). Switch the evalVersion option to `official` 
% for submissions.

  evalVersion = 'fast' ;

  pascal_models = {...
      'ssd-pascal-vggvd-300', ...
      'ssd-pascal-vggvd-512', ...
      'ssd-pascal-vggvd-ft-300', ...
      'ssd-pascal-vggvd-ft-512', ...
      'ssd-mcn-pascal-vggvd-300', ...
      'ssd-mcn-pascal-vggvd-512', ...
      'ssd-pascal-mobilenet-ft-300' ...
  } ;

  for ii = 1:numel(pascal_models)
      model = pascal_models{ii} ;     
      ssd_pascal_evaluation('modelName', model, 'evalVersion', evalVersion, ...
                            'refreshCache', 0, 'gpus', [3 4]) ;
  end

  coco_models = {...
   'ssd-mscoco-vggvd-300', ...
   'ssd-mscoco-vggvd-512', ...
   } ;

  for ii = 1:numel(coco_models)
      model = coco_models{ii} ;     
      ssd_coco_evaluation('modelName', model, ...
                          'refreshCache', 0, 'testset', 'val', ...
                          'useMiniVal', 1, 'gpus', [3 4]) ;
  end

