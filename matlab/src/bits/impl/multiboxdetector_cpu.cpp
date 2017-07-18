// @file multiboxdetector_cpu.cu
// @brief Multibox Detector CPU implementation, based 
// on Wei Liu's SSD caffe code
// @author Samuel Albanie
// @author Andrea Vedaldi

/*
Copyright (C) 2017- Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "multiboxdetector.hpp"
#include <bits/data.hpp>
#include <assert.h>
#include <float.h>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <string.h>

#include <map>
#include <vector>

/* ------------------------------------------------------------ */
/*                                                       useful */
/* ------------------------------------------------------------ */

template <typename T>
bool sortScorePairDescend(const std::pair<float, T>& pairA,
                          const std::pair<float, T>& pairB) {
    return pairA.first > pairB.first ;
}

typedef struct NormalizedBox {
    float xmin ;
    float ymin ;
    float xmax ;
    float ymax ;
    int label ;
    bool difficult ;
    float score ;
    float size ;
} NormalizedBox ;

typedef std::vector<NormalizedBox> BoxVector ;
typedef std::map<int, BoxVector> Label2BoxesMap ;
typedef std::map<int, std::vector<float> > Label2ScoreMap ;
typedef std::map<int, std::vector<int> > Label2PriorIndexMap ;
typedef std::map<int, std::vector<int> >::iterator indexIter ;

void getIntersection(const NormalizedBox &boxA, 
                     const NormalizedBox &boxB,
                     NormalizedBox* intersection) 
{
    // The intersection is computed as a box
    if (boxB.xmin > boxA.xmax || boxB.xmax < boxA.xmin ||
        boxB.ymin > boxA.ymax || boxB.ymax < boxA.ymin) {
      // Return [0, 0, 0, 0] if there is no intersection.
      intersection->xmin = 0 ;
      intersection->ymin = 0 ;
      intersection->xmax = 0 ;
      intersection->ymax = 0 ;
    } else {
      intersection->xmin = std::max(boxA.xmin, boxB.xmin) ;
      intersection->ymin = std::max(boxA.ymin, boxB.ymin) ;
      intersection->xmax = std::min(boxA.xmax, boxB.xmax) ;
      intersection->ymax = std::min(boxA.ymax, boxB.ymax) ;
    }
}

float getBoxSize(const NormalizedBox box) 
{
    float width = box.xmax - box.xmin ;
    float height = box.ymax - box.ymin ;
    if (width < 0 || height < 0) {
        return 0 ;
    } else {
        return width * height ;
    }
}

void getMaxScoreIndex(const std::vector<float>& scores, 
                      const float thresh,
                      const int topK, 
                      std::vector<std::pair<float, int>> *scoreIndexPairs) 
{
    // generate index score pairs for sufficiently high scores
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > thresh) {
            scoreIndexPairs->push_back(std::make_pair(scores[i], i)) ;
        }
    }

    // sort the score pair according to the scores in descending order
    std::stable_sort(scoreIndexPairs->begin(), scoreIndexPairs->end(), 
                                       sortScorePairDescend<int>) ;

    //printf("num boxes after conf threshold: %d \n", scoreIndexPairs->size()) ;
    // Keep top k scores if needed.
    if (topK > -1 && topK < scoreIndexPairs->size()) {
      scoreIndexPairs->resize(topK) ;
    }
    //printf("num boxes after conf threshold & topk: %d \n", scoreIndexPairs->size()) ;
}

float jaccardOverlap(const NormalizedBox &boxA, 
                     const NormalizedBox &boxB) 
{
    NormalizedBox intersection;
    getIntersection(boxA, boxB, &intersection);
    float width = intersection.xmax - intersection.xmin ;
    float height = intersection.ymax - intersection.ymin ;
    if (width > 0 && height > 0) {
        float intersectArea = width * height;
        float unionArea = getBoxSize(boxA) + getBoxSize(boxB) - intersectArea ;
        return intersectArea / unionArea ;
    } else {
        return 0.;
    }
}

void applyFastNMS(const BoxVector &boxes,
                  const std::vector<float>& scores, 
                  const float confThresh,
                  const float nmsThresh, 
                  const int keepTopK,
                  std::vector<int> *indices) 
{
    assert(boxes.size() == scores.size()) ;

    // retrieve top k scores (with corresponding indices).
    std::vector<std::pair<float, int>> scoreIndexPairs ;
    getMaxScoreIndex(scores, confThresh, keepTopK, &scoreIndexPairs) ;

    // run the nms - note we don't use adaptive NMS here
    int eta = 1 ;
    float adaptiveThresh = nmsThresh ;
    int numBoxes = boxes.size() ;
    //printf("num boxes total %d \n", numBoxes) ;
    //printf("num filtered boxes after thresh %d \n", scoreIndexPairs.size()) ;
    indices->clear() ;
    while (scoreIndexPairs.size() != 0) {
        const int idx = scoreIndexPairs.front().second ;
        bool keep = true ;
        for (int k = 0 ; k < indices->size() ; ++k) {
            if (keep) {
                const int keptIdx = (*indices)[k] ;
                float overlap = jaccardOverlap(boxes[idx], boxes[keptIdx]) ;
                //printf("overlap: %f \n", overlap) ;
                keep = overlap <= adaptiveThresh ;
            } else {
                break ;
            }
        }
        if (keep) {
            indices->push_back(idx) ;
        }
        scoreIndexPairs.erase(scoreIndexPairs.begin()) ;
        if (keep && eta < 1 && adaptiveThresh > 0.5) {
            adaptiveThresh *= eta ;
        }
    }
    //printf("num passing %d \n", indices->size()) ;
}

template <typename T>
void getConfScores(const T* confData, 
                   const int batchSize,
                   const int numPriors, 
                   const int numClasses,
                   std::vector<std::map<int, std::vector<float>>>* confPreds) 
{
    confPreds->clear();
    confPreds->resize(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        std::map<int, std::vector<float>> &labelScores = (*confPreds)[i];
        for (int p = 0; p < numPriors; ++p) {
            int startIdx = p * numClasses;
            for (int c = 0; c < numClasses; ++c) {
                // MATLAB is column major ....
                //labelScores[c].push_back(confData[numPriors * c + p]);
                labelScores[c].push_back(confData[startIdx + c]);
            }
        }
        confData += numPriors * numClasses;
    }
    //printf("score batch 1 %f\n", (*confPreds)[0][0][0]) ;
    //printf("score batch 2 %f\n", (*confPreds)[1][0][0]) ;
}

template<typename T>
void getLocPreds(const T* locData, 
                 const int batchSize, 
                 const int numPriors, 
                 std::vector<std::map<int, std::vector<NormalizedBox>>> *locPreds) 
{
    locPreds->clear() ;
    locPreds->resize(batchSize) ;
    for (int i = 0 ; i < batchSize ; i++) {
        Label2BoxesMap &labelBox = (*locPreds)[i] ;
        for (int p = 0; p < numPriors; ++p) {
            int startIdx = p * 4 ;
            int label = -1 ; 
            if (labelBox.find(label) == labelBox.end()) {
              labelBox[label].resize(numPriors) ;
            }
            labelBox[label][p].xmin = locData[startIdx];
            labelBox[label][p].ymin = locData[startIdx + 1] ;
            labelBox[label][p].xmax = locData[startIdx + 2] ;
            labelBox[label][p].ymax = locData[startIdx + 3] ;
        }
        locData += numPriors * 4 ;
    }
}

template <typename T>
void getPriorBoxes(const T* priors, 
                   const int numPriors,
                   BoxVector* priorBoxes,
                   std::vector<std::vector<float>>* priorVars) 
{
    priorBoxes->clear() ;
    priorVars->clear() ;
    for (int p = 0; p < numPriors; ++p) {
        int startIdx = p * 4 ;
        NormalizedBox box ;
        box.xmin = priors[startIdx] ;
        box.ymin = priors[startIdx + 1] ;
        box.xmax = priors[startIdx + 2] ;
        box.ymax = priors[startIdx + 3] ;
        box.size = getBoxSize(box) ;
        priorBoxes->push_back(box) ;
    }

    for (int p = 0; p < numPriors; ++p) {
        int startIdx = (numPriors + p) * 4;
        std::vector<float> var;
        for (int j = 0; j < 4; ++j) {
            var.push_back(priors[startIdx + j]);
        }
        priorVars->push_back(var);
    }
}

void decodeBox(const NormalizedBox& priorBox, 
               const std::vector<float>& priorVar,
               const NormalizedBox& box,
               NormalizedBox* decodedBox) 
{
    float priorWidth = priorBox.xmax - priorBox.xmin ;
    float priorHeight = priorBox.ymax - priorBox.ymin ;
    float priorCenterX = (priorBox.xmin + priorBox.xmax) / 2. ;
    float priorCenterY = (priorBox.ymin + priorBox.ymax) / 2.;
    assert(priorWidth > 0) ;
    assert(priorHeight > 0) ;

    float decodedCenterX, decodedCenterY, decodedWidth, decodedHeight ;
    decodedCenterX = priorVar[0] * box.xmin * priorWidth + priorCenterX ;
    decodedCenterY = priorVar[1] * box.ymin * priorHeight + priorCenterY ;
    decodedWidth = exp(priorVar[2] * box.xmax) * priorWidth ;
    decodedHeight = exp(priorVar[3] * box.ymax) * priorHeight ;

    decodedBox->xmin = (decodedCenterX - decodedWidth / 2.) ;
    decodedBox->ymin = (decodedCenterY - decodedHeight / 2.) ;
    decodedBox->xmax = (decodedCenterX + decodedWidth / 2.) ;
    decodedBox->ymax = (decodedCenterY + decodedHeight / 2.) ;
    decodedBox->size = getBoxSize(*decodedBox) ;
}

void decodeBoxes(const BoxVector &priorBoxes,
                 const std::vector<std::vector<float>>& priorVars,
                 const BoxVector& boxes,
                 BoxVector* decodedBoxes) 
{
    assert(priorBoxes.size() == priorVars.size()) ;
    assert(priorBoxes.size() == boxes.size()) ;
    int numBoxes = priorBoxes.size();
    if (numBoxes >= 1) {
        assert(priorVars[0].size() == 4) ;
    }
    decodedBoxes->clear();
     for (int i = 0; i < numBoxes; ++i) {
        NormalizedBox decodedBox;
        decodeBox(priorBoxes[i], priorVars[i], boxes[i], &decodedBox);
        decodedBoxes->push_back(decodedBox);
     }
}

void decodeBoxesBatch(const std::vector<Label2BoxesMap>& locPredsBatch,
                 const BoxVector &priorBoxes,
                 const std::vector<std::vector<float> >& priorVars,
                 const int batchSize, 
                 std::vector<Label2BoxesMap> *decodedBoxesBatch) 
{
    assert(locPredsBatch.size() == batchSize) ;
    decodedBoxesBatch->clear() ;
    decodedBoxesBatch->resize(batchSize) ;
    for (int i = 0 ; i < batchSize ; ++i) {
        // Decode predictions into bboxes.
        Label2BoxesMap& decodedBoxes = (*decodedBoxesBatch)[i] ;
        int label = -1 ;
        if (locPredsBatch[i].find(label) == locPredsBatch[i].end()) {
            printf("Unable to find location preds for label %d", label) ;
        }
        const BoxVector &labelLocPreds = locPredsBatch[i].find(label)->second ;
        decodeBoxes(priorBoxes, priorVars, labelLocPreds, &(decodedBoxes[label])) ;
    }
    //printf("decoded box for label %d xmin: %f\n", (*decodedBoxesBatch)[0][-1][0].xmin) ;
    //printf("decoded box for label %d ymin: %f\n", (*decodedBoxesBatch)[0][-1][0].ymin) ;
    //printf("decoded box for label %d xmax: %f\n", (*decodedBoxesBatch)[0][-1][0].xmax) ;
    //printf("decoded box for label %d ymax: %f\n", (*decodedBoxesBatch)[0][-1][0].ymax) ;
}

namespace vl { namespace impl {

  template<typename T>
  struct multiboxdetector<vl::VLDT_CPU,T>
  {

    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T const* locPreds,
            T const* confPreds,
            T const* priors,
            int nmsTopK,
            int keepTopK,
            int numClasses,
            float nmsThresh,
            float confThresh, 
            int backgroundLabel, 
            size_t outHeight, 
            size_t outWidth, 
            size_t batchSize, 
            size_t numPriors) 
    {
      //printf("cpu version running\n") ;
      //printf("vl_multiboxdetector: nmsTopK: %d\n", nmsTopK) ;
      //printf("vl_multiboxdetector: keepTopK: %d\n", keepTopK) ;
      //printf("vl_multiboxdetector: numClasses: %d\n", numClasses) ;
      //printf("vl_multiboxdetector: nmsThresh: %f\n", nmsThresh) ;
      //printf("vl_multiboxdetector: confThresh: %f\n", confThresh) ;
      //printf("vl_multiboxdetector: backgroundLabel: %d\n", backgroundLabel) ;
      //printf("vl_multiboxdetector: outHeight: %d\n", outHeight) ;
      //printf("vl_multiboxdetector: outWidth: %d\n", outWidth) ;
      //printf("vl_multiboxdetector: batchSize: %d\n", batchSize) ;
      //printf("vl_multiboxdetector: num priors: %d\n", numPriors) ;

      std::vector<Label2BoxesMap> locPredsBatch ;
      getLocPreds(locPreds, batchSize, numPriors, &locPredsBatch) ;

      // TODO: Check indexing order (over priors vs over classes)
      std::vector<Label2ScoreMap> confScoresBatch ;
      getConfScores(confPreds, batchSize, numPriors, numClasses, &confScoresBatch) ;

      // Retrieve all prior boxes. It is the same within a batch since we 
      // assume all images in a batch are of same dimension.
      BoxVector priorBoxes ;
      std::vector<std::vector<float> > priorVars ;
      getPriorBoxes(priors, numPriors, &priorBoxes, &priorVars) ;


      // Decode all location predictions to boxes.
      std::vector<Label2BoxesMap> decodedBoxesBatch ;
      decodeBoxesBatch(locPredsBatch, priorBoxes, priorVars, 
                                        batchSize, &decodedBoxesBatch) ;

      int numKept = 0 ;
      std::vector<Label2PriorIndexMap> batchIndices ;
      for (int i = 0 ; i < batchSize ; ++i) {
          const Label2BoxesMap& decodedBoxes = decodedBoxesBatch[i] ;
          const Label2ScoreMap &confScores = confScoresBatch[i] ;
          Label2PriorIndexMap indices ;
          int numDetections = 0 ;

          // boxes are shared so we just retrieve them once
          int label = -1 ;
          if (decodedBoxes.find(label) == decodedBoxes.end()) {
              printf("Could not find loc predictions for label %d", label) ;
          }
          const BoxVector &boxes = decodedBoxes.find(label)->second ;

          for (int c = 0 ; c < numClasses ; ++c) {
              if ((c + 1) == backgroundLabel) { // +1 for MATLAB offset 
                  // Ignore background class.
                  continue ;
              }
              if (confScores.find(c) == confScores.end()) {
                  printf("Could not find conf predictions for label %d", c) ;
              }
              const std::vector<float> &scores = confScores.find(c)->second ;
              applyFastNMS(boxes, scores, confThresh, nmsThresh, nmsTopK, &(indices[c])) ;
              numDetections += indices[c].size();
          }
          if (keepTopK > -1 && numDetections > keepTopK) {
              std::vector<std::pair<float, std::pair<int, int>>> scoreIndexPairs ;
              for (indexIter it = indices.begin() ; it != indices.end() ; ++it) {
                  int label = it->first ;
                  const std::vector<int> &labelIndices = it->second ;
                  if (confScores.find(label) == confScores.end()) {
                      // Something bad happened for current label.
                      printf("Could not find location predictions for %d", label) ;
                      continue;
                  }
                  const std::vector<float> &scores = confScores.find(label)->second ;
                  for (int j = 0 ; j < labelIndices.size() ; ++j) {
                      int idx = labelIndices[j] ;
                      assert(idx < scores.size()) ;
                      scoreIndexPairs.push_back(
                         std::make_pair(scores[idx], std::make_pair(label, idx))) ;
                  }
              }
              // Keep top k results per image.
              std::sort(scoreIndexPairs.begin(), 
                        scoreIndexPairs.end(),
                        sortScorePairDescend<std::pair<int, int>>) ;
              scoreIndexPairs.resize(keepTopK) ;

              // Store the new indices.
              Label2PriorIndexMap newIndices;
              for (int j = 0 ; j < scoreIndexPairs.size() ; ++j) {
                  int label = scoreIndexPairs[j].second.first ;
                  int idx = scoreIndexPairs[j].second.second ;
                  newIndices[label].push_back(idx) ;
              }
              batchIndices.push_back(newIndices) ;
              numKept += keepTopK ;
          } else {
              batchIndices.push_back(indices) ;
              numKept += numDetections ;
          }
      }

      for (int i = 0 ; i < batchSize ; ++i) {
          int count = 0 ; // fixed size outputs
          const Label2ScoreMap &confScores = confScoresBatch[i] ;
          const Label2BoxesMap &decodedBoxes = decodedBoxesBatch[i] ;

          for (indexIter it = batchIndices[i].begin() ; 
                                 it != batchIndices[i].end(); ++it) {
              int label = it->first ;
              if (confScores.find(label) == confScores.end()) {
                  printf("Could not find conf preds for %d", label) ;
                  continue;
              }
              const std::vector<float> &scores = confScores.find(label)->second ;
              int locLabel = -1 ;
              if (decodedBoxes.find(locLabel) == decodedBoxes.end()) {
                  printf("Could not find loc preds for %d", locLabel) ;
                  continue;
              }
              const BoxVector &boxes = decodedBoxes.find(locLabel)->second ;
              std::vector<int> &indices = it->second ;

              int numIndices = indices.size() ;
              for (int j = 0 ; j < numIndices ; ++j) {
                  int idx = indices[j] ;
                  const NormalizedBox& box = boxes[idx] ;
                  output[outHeight * i * 6 + count ] = label + 1 ; // MATLAB +1
                  output[outHeight * i * 6 + outHeight + 1 * count] = scores[idx] ;
                  output[outHeight * i * 6 + outHeight * 2 + count] = box.xmin ;
                  output[outHeight * i * 6 + outHeight * 3 + count] = box.ymin ;
                  output[outHeight * i * 6 + outHeight * 4 + count] = box.xmax ;
                  output[outHeight * i * 6 + outHeight * 5 + count] = box.ymax ;
                ++count;
              }
          }
      }
      return VLE_Success ;
   }
 } ;
} } // namespace vl::impl

template struct vl::impl::multiboxdetector<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::multiboxdetector<vl::VLDT_CPU, double> ;
#endif
