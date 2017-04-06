%VL_NNMULTIBOXDETECTOR produces SSD detections
%   Y = VL_NNMULTIBOXDETECTOR(L, C, P) produces a set of bounding boxes
%   and class confidence scores from the network predictions in 
%   multiple steps.  Firstly, it perfoms a decoding task, by 
%   taking netowrk predictions that have been made *relative* 
%   to a set of prior boxes and translates them into predictions 
%   that lie in the image space.  Predictions with a confidence
%   that exceed a given threshold are then passed into a Non 
%   Maximum Supression (NMS) step.  In the following, `N` denotes 
%   the batch size:
%
%     L is a 1 x 1 x C1 x N array containing the
%         location predictions of the network (these
%         predictions take the form of encoded spatial
%         updates to the prior boxes), where N is the
%         batch size and C1 = 4 * numPriorBoxes
%
%     C is a 1 x 1 x C2 x N array containing the
%         per-class confidence predictions of the network,
%         where C2 = numClasses * numPriorBoxes
%
%     P is a C3 x 1 x 2 x N array containing the
%         prior boxes, which are encoded as a set of bbox
%         coordinates (xminx, ymin, xmax, ymax) and a set of
%         four "variances" which are used to scale the resulting
%         boxes, where C3 = 4 * numPriorBoxes
%
%     Y is a D x 26 x 1 x N array where D is the number of
%         detections and the first column is the predicted class
%         label, the next four columns contain the predicted
%         [xmin, ymin, xmax, ymax] and the remaining columns are
%         the scores across each of the 21 classes
%
%   VL_NNMULTIBOXDETECTOR(...,'OPT',VALUE,...) takes the following options:
%
%   `numClasses`:: 21
%    The number of classes the network aims to predict (corresponds
%    to the number of confidence predictions) 
%
%   `nmsThreshold`:: 0.45
%    The (usual) threshold used in non maximum supression
%
%   `nmsTopK`:: 400
%    The maximum number of bounding boxes that are passed into the 
%    NMS algorithm, arranged in descending order. 
%
%   `confidenceThreshold`:: 0.01
%    Defines a minimum confidence score a prediction must have to be 
%    considered for NMS.
%
%   `keepTopK`:: 200
%    Maximum number of predictions to be kept per image after NMS
