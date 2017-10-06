% This script evaluates the pre-trained models released by the 
% mastermind behind the SSD detector, Wei Liu, as well as some models
% trained with the matconvnet implementation.  It also evaluates the 
% mobilenet detector released by chuanqi305 on GitHub.
%
% Some notes about the pretrained models/Pascal VOC evaluation
% -------------------------------------------------------------
%
% The evaluation is done on the VOC 2007 test data. 
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

models = {...
    'ssd-pascal-vggvd-300', ...
    'ssd-pascal-vggvd-512', ...
    'ssd-pascal-vggvd-ft-300', ...
    'ssd-pascal-vggvd-ft-512', ...
    'ssd-mcn-pascal-vggvd-300', ...
    'ssd-mcn-pascal-vggvd-512', ...
    'ssd-pascal-mobilenet-ft' ...
} ;

% minus imagenet mean: 66.5
% minus 127.5:  67.1

for i = 1:numel(models)
    model = models{i} ; args = {} ;

    % The MobileNet detector requires different input preprocessing
    if contains(model, 'mobilenet')
      args = {'batchOpts', ...
              struct('imMean', [127.5, 127.5, 127.5], ... %[123, 117, 104], ...
              'scaleInputs', 0.007843)} ;
%[127.5 127.5 127.5], ...
    end

    ssd_pascal_evaluation('modelName', model, ...
                          'evalVersion', evalVersion, args{:}) ;
end

