function res = eval_voc(cls, image_ids, bboxes, scores, VOCopts, varargin)
% res = eval_voc(cls, boxes, imdb, suffix)
%   Use the VOCdevkit to evaluate detections specified in boxes
%   for class cls against the ground-truth boxes in the image
%   database imdb. Results files are saved with an optional
%   suffix.

% This is a modified version of Ross Girshick's code 
% released as part of R-CNN. 
% 
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Add a random string ("salt") to the end of the results file name
% to prevent concurrent evaluations from clobbering each other
use_res_salt = true;

% Delete results files after computing APs
rm_res = true;

% comp4 because we use outside data (ILSVRC2012)
comp_id = 'comp4';

% draw each class curve
drawCurve = true;

% save results
opts.suffix = '' ;

% the official eval is quite slow, so use x10 version for development
% and switch to official for paper experiments
opts.evalVersion = '2007' ;

[opts, varargin] = vl_argparse(opts, varargin) ;

if ~strcmp(opts.suffix, '')
    suffix = ['_' suffix] ;
end

if use_res_salt
  prev_rng = rng;
  rng shuffle;
  salt = sprintf('%d', randi(100000));
  res_id = [comp_id '-' salt];
  rng(prev_rng);
else
  res_id = comp_id;
end

res_fn = sprintf(VOCopts.detrespath, res_id, cls);

% write out detections in PASCAL format and score
fid = fopen(res_fn, 'w');
for i = 1:numel(image_ids);
  fprintf(fid, '%s %f %.3f %.3f %.3f %.3f\n', image_ids{i}, scores(i), bboxes(i,:));
end
fclose(fid);

% Bug in VOCevaldet requires that tic has been called first
tic;
switch opts.evalVersion
    case '2007'
        [recall, prec, ap] = VOCevaldet(VOCopts, res_id, cls, drawCurve);
    case '2010'
        [recall, prec ap] = x10VOCevaldet(VOCopts, res_id, cls, drawCurve);
    otherwise
        error(sprintf('Evaluation version %s not recognised', opts.evalVersion)) ;
end
ap_auc = xVOCap(recall, prec);

% force plot limits
ylim([0 1]);
xlim([0 1]);

print(gcf, '-djpeg', '-r0', ...
    fullfile(VOCopts.cacheDir, sprintf('%s_pr_%s.jpg', cls, opts.suffix))) ;
fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc) ;

save(fullfile(VOCopts.cacheDir,  sprintf('%s_pr_%s',cls, opts.suffix)), ...
    'recall', 'prec', 'ap', 'ap_auc') ;

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;
if rm_res
  delete(res_fn);
end

% -----------------------------
function ap = xVOCap(rec,prec)
% -----------------------------
% this function is part of the PASCAL VOC 2011 devkit

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
