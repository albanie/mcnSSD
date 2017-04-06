function newBoxes = bboxCoder(boxes, from, to)
%BBOXCODER provides alternative boudning box encodings
% newBoxes = BBOXCODER(BOXES) re-enocdes an H x 4 x 1 x N
% array of bounding boxes (where N is the batch size and 
% H is the number of boxes per batch) in one of the named 
% formats given below, and re-encodes it to another format 
% with identical dimensions.
%
%   Available formats
%   'MinMax': [XMIN YMIN XMAX YMAX]
%   'MinWH': [XMIN YMIN WIDTH HEIGHT]
%   'CenWH': [CENTERX CENTERY WIDTH HEIGHT]
%
% Example:
%    newBoxes = bboxCoder(bboxes, 'MinMax', 'CenWH') 

assert(~strcmp(from,to), '`from` and `to` should be different encodings') ;

switch from
    case 'MinMax'
        WH = boxes(:,3:4,:,:) - boxes(:,1:2,:,:) ;
        if strcmp(to, 'CenWH')
            CenXY = boxes(:,1:2,:,:) + WH / 2 ;
        else
            MinXY = boxes(:,1:2,:,:) ;
        end
    case 'MinWH'
        WH = boxes(:,3:4,:,:) ;
        if strcmp(to, 'CenWH')
            CenXY = boxes(:,1:2,:,:) + WH / 2 ;
        else
            MinXY = boxes(:,1:2,:,:) ;
            MaxXY = MinXY + WH ;
        end

    case 'CenWH'
        WH = boxes(:,3:4,:,:) ;
        MinXY = boxes(:,1:2,:,:) - WH / 2;
        if strcmp(to, 'MinMax')
            MaxXY = MinXY + WH;
        end
    otherwise
        fprintf('%s is not a supported encoding', from) ;
end

switch to
    case 'MinMax'
        newBoxes = [ MinXY MaxXY] ;
    case 'MinWH'
        newBoxes = [ MinXY WH] ;
    case 'CenWH'
        newBoxes = [ CenXY WH] ;
    otherwise
        fprintf('%s is not a supported encoding', to) ;
end
