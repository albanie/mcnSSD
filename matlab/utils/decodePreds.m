function bboxes = decodePreds(preds, priorBoxes, priorVars) 
%DEOCDEPREDS 
% TODO: docs
% returns boxes in [xmin, ymin, xmax, ymax] format

priorBoxesCenterWH = bboxCoder(priorBoxes, 'MinMax', 'CenWH') ;

% Decode bounding box predictions 
predsXY = preds(:, 1:2) .* priorBoxesCenterWH(:, 3:4) ...
            .* priorVars(:,1:2) + priorBoxesCenterWH(:,1:2) ;
predsWH = exp(preds(:, 3:4) .* priorVars(:, 3:4)) ...
                        .* priorBoxesCenterWH(:,3:4) ;
bboxes = [ predsXY - predsWH / 2 predsXY + predsWH / 2 ] ;
