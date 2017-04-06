% This script evaluates the pre-trained models released by the 
% mastermind behind the SSD detector, Wei Liu

% Setup MatConvNet
run(fullfile(fileparts(mfilename('fullpath')), ...
                  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(vl_rootnn, 'examples/ssd')) ;

evalVersion = '2007' ;

models = {...
    'ssd-pascal-vggvd-300', ...
    'ssd-pascal-vggvd-512', ...
    'ssd-pascal-plus-vggvd-300', ...
    'ssd-pascal-vggvd-ft-300', ...
    'ssd-pascal-plus-vggvd-512', ...
    'ssd-pascal-plus-vggvd-ft-300', ...
    'ssd-pascal-plus-vggvd-ft-512', ...
    'ssd-pascal-vggvd-ft-512', ...
} ;

for i = 1:numel(models)
    model = models{i} ;
    ssd_pascal_evaluation('modelName', model, ...
                          'evalVersion', evalVersion) ;
end

% Some notes about the pretrained models/Pascal VOC evaluation
% -------------------------------------------------------------
%
% The evaluation is done on the VOC 2007 test data. 
%
% There are two official evaluation scripts. Pre-2010, the average 
% precision was computed at a set of fixed interval points.  From 2010, 
% the computation was changed to perform numerical integration across 
% the whole curve (details of the change
% can be found here (Section 3.4.1): 
% http://host.robots.ox.ac.uk/pascal/VOC/voc2010/htmldoc/devkit_doc.html

% The two versions produce slightly different scores. Use the 2007 version
% for computing official results, but while developing networks the 2010 
% script is useful since it computes the scores significantly faster. 
% To give an idea of the difference in scores, the performance of the models
% under each evaluation is given below.
%
%
% Test Set Results (2007 evaluation script):
% -------------- -------------------- --------------------
%                ssd-pascal-vggvd-300 ssd-pascal-vggvd-512
% -------------- -------------------- --------------------
% aeroplane      80.53                85.36
% bicycle        83.77                85.74
% bird           76.40                81.07
% boat           71.53                73.01
% bottle         50.17                58.09
% bus            86.90                87.91
% car            86.05                88.43
% cat            88.57                87.50
% chair          59.96                63.90
% cow            81.39                85.46
% diningtable    76.30                73.10
% dog            85.92                86.21
% horse          86.60                86.99
% mean           77.54                79.57
% motorbike      83.62                83.89
% person         79.57                82.69
% pottedplant    52.62                55.06
% sheep          79.22                80.93
% sofa           78.89                79.27
% train          86.52                86.61
% tvmonitor      76.31                80.17
% -------------- -------------------- --------------------
% mean           77.54                79.57
% -------------- -------------------  --------------------
%
%  Test Set Results (faster 2010 evaluation script):
% ------------ --------------------- --------------------
%               ssd-pascal-vggvd-300 ssd-pascal-vggvd-512
% ------------ --------------------- --------------------
% aeroplane      81.82                87.99
% bicycle        87.12                89.68
% bird           78.83                85.01
% boat           71.48                75.18
% bottle         50.23                58.74
% bus            89.05                91.96
% car            88.82                92.98
% cat            91.97                92.27
% chair          61.08                65.62
% cow            84.05                88.93
% diningtable    79.05                75.71
% dog            88.43                90.66
% horse          89.38                90.81
% mean           79.76                82.80
% motorbike      86.55                87.71
% person         82.19                85.64
% pottedplant    53.36                56.68
% sheep          80.07                84.59
% sofa           82.41                82.60
% train          90.04                89.74
% tvmonitor      79.29                83.49
% -------------- -------------------- --------------------
% mean           79.76                82.80
% -------------- -------------------- --------------------
