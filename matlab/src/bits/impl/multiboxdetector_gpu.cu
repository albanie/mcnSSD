// @file multiboxdetector_gpu.cu
// @brief Multibox Detector GPU implementation, 
// based on Wei Liu's SSD caffe code
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
/*                                                      kernels */
/* ------------------------------------------------------------ */

enum {
  XMIN = 0,
  YMIN,
  XMAX,
  YMAX,
} ;


template <typename T>
__global__ void decodeBoxesKernel(const int numThreads,
                                  const int numPriors, 
                                  const T* locPreds, 
                                  const T* priors,
                                  T* decodedBoxes) 
{
    // Grid stride-loop 
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
             index < numThreads ; 
             index += blockDim.x * gridDim.x) 
    {
        const int modIndex = index % 4 ;
        const int priorIndex = ((index / 4 ) % numPriors) * 4 ;
        const int varIndex = priorIndex + numPriors * 4 ;

        if ((modIndex == XMIN) || (modIndex == XMAX)) {

            const T priorXmin = priors[priorIndex] ;
            const T priorXmax = priors[priorIndex + 2] ;
            const T priorWidth = priorXmax - priorXmin;
            const T priorCenterX = (priorXmin + priorXmax) / 2.;
            const T xmin = locPreds[index - modIndex];
            const T xmax = locPreds[index - modIndex + 2];

            T decodedWidth ; 
            T decodedCenterX ; 

            decodedCenterX = priors[varIndex] * xmin * priorWidth + priorCenterX ;
            decodedWidth = exp(priors[varIndex + 2] * xmax) * priorWidth ;

            if (modIndex == XMIN) {
                decodedBoxes[index] = decodedCenterX - decodedWidth / 2.;
            } else {
                decodedBoxes[index] = decodedCenterX + decodedWidth / 2.;
            }

        } else {

            const T priorYmin = priors[priorIndex + 1] ;
            const T priorYmax = priors[priorIndex + 3] ;
            const T priorHeight = priorYmax - priorYmin;
            const T priorCenterY = (priorYmin + priorYmax) / 2.;
            const T ymin = locPreds[index - modIndex + 1];
            const T ymax = locPreds[index - modIndex + 3];

            T decodedHeight ;
            T decodedCenterY ;

            decodedCenterY = priors[varIndex + 1] * ymin * priorHeight + priorCenterY ;
            decodedHeight = exp(priors[varIndex + 3] * ymax) * priorHeight ;

            if (modIndex == YMIN) {
                decodedBoxes[index] = decodedCenterY - decodedHeight / 2.;
            } else {
                decodedBoxes[index] = decodedCenterY + decodedHeight / 2.;
            }
        }
    }
}

template <typename T>
__global__ void permuteConfsKernel(const int numThreads,
                                   const int numClasses, 
                                   const int numPriors,
                                   const T* confPreds, 
                                   T* permuted) 
{
    // Grid stride-loop 
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
             index < numThreads ; 
             index += blockDim.x * gridDim.x) 
    {
        const int classIndex = index % numClasses;
        const int priorIndex = (index / numClasses) % numPriors;
        const int batchIndex = (index / numClasses) / numPriors;
        const int newIndex = (batchIndex * numClasses + classIndex) * numPriors + priorIndex ;
        permuted[newIndex] = confPreds[index];
    }
}

/* ------------------------------------------------------------ */
/*                                              kernel wrappers */
/* ------------------------------------------------------------ */

template <typename T>
void decodeBoxesGPU(const int numThreads,
                    const int numPriors, 
                    const T* locPreds, 
                    const T* priors,
                    T* decodedBoxes) 
{
    int numBlocks = (numThreads + 511) / 512 ;
    decodeBoxesKernel<T><<<numBlocks, 512>>>(numThreads,  
                                             numPriors, 
                                             locPreds, 
                                             priors, 
                                             decodedBoxes) ;
    cudaError_t status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        exit(-1) ;
    }
}

template <typename T>
void permuteConfsGPU(const int numThreads,
                    const int numClasses, 
                    const int numConfPreds,
                    const T* confPreds, 
                    T* permuted) 
{
    int numBlocks = (numThreads + 511) / 512 ;
    permuteConfsKernel<T><<<numBlocks, 512>>>(numThreads, 
                                              numClasses, 
                                              numConfPreds,
                                              confPreds, 
                                              permuted);
    cudaError_t status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        exit(-1) ;
    }
}  

/* ------------------------------------------------------------ */
/*                                             non kernel utils */
/* ------------------------------------------------------------ */

template <typename T>
bool sortScorePairDescend(const std::pair<float, T>& pairA,
                          const std::pair<float, T>& pairB) 
{
    return pairA.first > pairB.first ;
}

template <typename T>
T getBoxSize(const T* box) {
    T boxSize ;
    if (box[2] < box[0] || box[3] < box[1]) {
        // return 0 for invalid boxes
        boxSize = T(0.) ;
    } else {
        const T width = box[2] - box[0];
        const T height = box[3] - box[1];
        boxSize = width * height;
    }
    return boxSize ;
}

template <typename T>
T jaccardOverlap(const T* boxA, const T* boxB) 
{
    if (boxB[0] > boxA[2] || 
        boxB[2] < boxA[0] ||
        boxB[1] > boxA[3] || 
        boxB[3] < boxA[1]) {
        // return 0 for invalid boxes
        return T(0.) ;
    } else {
        const T xminIntersection = std::max(boxA[0], boxB[0]) ;
        const T yminIntersection = std::max(boxA[1], boxB[1]) ;
        const T xmaxIntersection = std::min(boxA[2], boxB[2]) ;
        const T ymaxIntersection = std::min(boxA[3], boxB[3]) ;

        const T widthIntersection = xmaxIntersection - xminIntersection;
        const T heightIntersection = ymaxIntersection - yminIntersection;
        const T sizeIntersection = widthIntersection * heightIntersection;

        const T sizeBoxA = getBoxSize(boxA);
        const T sizeBoxB = getBoxSize(boxB);

        return sizeIntersection / (sizeBoxA + sizeBoxB - sizeIntersection);
    }
}

template <typename T>
void getMaxScoreIndexCPU(const T* scores, 
                         const float thresh,
                         const int numPriors,
                         const int topK, 
                         std::vector<std::pair<float, int> > *scoreIndexPairs) 
{
    // generate index score pairs for sufficiently high scores
    for (int i = 0 ; i < numPriors ; ++i) {
        if (scores[i] > thresh) {
            scoreIndexPairs->push_back(std::make_pair(scores[i], i)) ;
        }
    }

    // sort the score pair according to the scores in descending order
    std::stable_sort(scoreIndexPairs->begin(), scoreIndexPairs->end(), 
                                       sortScorePairDescend<int>) ;

    // Keep top k scores if needed.
    if (topK > -1 && topK < scoreIndexPairs->size()) {
      scoreIndexPairs->resize(topK) ;
    }
}

template <typename T>
void applyFastNMSCPU(const T* boxes,
                     const T* scores, 
                     const float confThresh,
                     const float nmsThresh, 
                     const int numPriors,
                     const int keepTopK,
                     std::vector<int> *indices) 
{
    // retrieve top k scores (with corresponding indices).
    std::vector<std::pair<float, int> > scoreIndexPairs ;
    getMaxScoreIndexCPU(scores, confThresh, numPriors, keepTopK, &scoreIndexPairs) ;

    // run the nms - note we don't use adaptive NMS here
    int eta = 1 ;
    float adaptiveThresh = nmsThresh ;
    indices->clear() ;

    while (scoreIndexPairs.size() != 0) {
        const int idx = scoreIndexPairs.front().second ;
        bool keep = true ;
        int sz = indices->size() ;
        for (int k = 0 ; k < indices->size() ; ++k) {
            if (keep) {
                const int keptIdx = (*indices)[k] ;
                float overlap = jaccardOverlap(boxes + idx * 4, 
                                               boxes + keptIdx * 4) ;
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
}


/* ------------------------------------------------------------ */
/*                                                      forward */
/* ------------------------------------------------------------ */

namespace vl { namespace impl {

    template<typename T>
    struct multiboxdetector<vl::VLDT_GPU,T>
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
    // The first two steps of the forward pass are performed on the GPU i.e.
    //
    // 1. Decoding the box predictions
    // 2. Permuting the confidence scores
    //
    // Following this, the data is returned to the CPU and the NMS is run 
    // serially - this can be updated when we have time :)

    const int BOXES_ARRAY_SIZE = numPriors * 4 * batchSize ;
    const int BOXES_ARRAY_BYTES = BOXES_ARRAY_SIZE * sizeof(T) ;
    T * decodedBoxes ;
    cudaMalloc( (void **) &decodedBoxes, BOXES_ARRAY_BYTES) ;

    const int numLocPreds = numPriors * 4 * batchSize ;
    decodeBoxesGPU<T>(numLocPreds, 
                      numPriors, 
                      locPreds, 
                      priors, 
                      decodedBoxes) ;

    // permute the confidence predictions to allow contiguous access
    const int CONF_ARRAY_SIZE = numPriors * numClasses * batchSize ;
    const int CONF_ARRAY_BYTES = CONF_ARRAY_SIZE * sizeof(T) ;
    T * permutedConfPreds ;
    cudaMalloc( (void **) &permutedConfPreds, CONF_ARRAY_BYTES) ;

    const int numConfPreds = numPriors * numClasses * batchSize ;
    permuteConfsGPU<T>(numConfPreds, 
                       numClasses, 
                       numPriors, 
                       confPreds,
                       permutedConfPreds) ;

    // allocate on host, copy data back and free up the GPU
    T* h_decodedBoxes = new T [BOXES_ARRAY_SIZE] ;
    T*  h_permutedConfPreds = new T [CONF_ARRAY_SIZE] ;

    cudaMemcpy(h_decodedBoxes, decodedBoxes, 
               BOXES_ARRAY_BYTES, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(h_permutedConfPreds, permutedConfPreds, 
               CONF_ARRAY_BYTES, cudaMemcpyDeviceToHost) ;
    cudaFree(decodedBoxes) ;
    cudaFree(permutedConfPreds) ;

    int numKept = 0 ;
    std::vector<std::map<int, std::vector<int> > > batchIndices ;

    for (int i = 0; i < batchSize; ++i) {

        int numDetections = 0 ;
        std::map<int, std::vector<int> > indices ;
        int confIdxOffset = numClasses * numPriors * i ;
        int boxIdxOffset = numPriors * 4 * i ;

        for (int c = 0 ; c < numClasses ; ++c) {
          if ((c + 1) == backgroundLabel) { // ignore background (MATLAB indexing)
            continue ;
          }
          T* boxes_ = h_decodedBoxes + boxIdxOffset;
          T* confPreds_ = h_permutedConfPreds + confIdxOffset + c * numPriors ;


          applyFastNMSCPU(boxes_, 
                          confPreds_, 
                          confThresh, 
                          nmsThresh, 
                          numPriors, 
                          nmsTopK, 
                          &(indices[c])) ;
          numDetections += indices[c].size() ;
        }

        if (keepTopK > -1 && numDetections > keepTopK) {
            std::vector<std::pair<float, std::pair<int, int> > > scoreIndexPairs ;
            for (std::map<int, std::vector<int> >::iterator it = indices.begin() ;
                 it != indices.end(); ++it) {
                int label = it->first ;
                const std::vector<int>& labelIndices = it->second ;
                for (int j = 0; j < labelIndices.size(); ++j) {
                  int idx = labelIndices[j] ;
                  float score = h_permutedConfPreds[confIdxOffset + label * numPriors + idx] ;
                  scoreIndexPairs.push_back(std::make_pair(
                                         score, std::make_pair(label, idx))) ;
                }
            }

            // Keep top k results per image.
            std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
                      sortScorePairDescend<std::pair<int, int> >);
            scoreIndexPairs.resize(keepTopK);

            // Store the new indices.
            std::map<int, std::vector<int> > newIndices;
            for (int j = 0; j < scoreIndexPairs.size(); ++j) {
                int label = scoreIndexPairs[j].second.first;
                int idx = scoreIndexPairs[j].second.second;
                newIndices[label].push_back(idx);
            }
            batchIndices.push_back(newIndices);
            numKept += keepTopK;
          } else {
              batchIndices.push_back(indices);
              numKept += numDetections;
        }
    }

    for (int i = 0 ; i < batchSize ; ++i) {
        int count = 0 ; // fixed size outputs
        int boxIdxOffset = numPriors * 4 * i ;
        int confIdxOffset = numClasses * numPriors * i ;

        for (std::map<int, std::vector<int> >::iterator it 
             = batchIndices[i].begin() ; it != batchIndices[i].end() ; ++it) {

            int label = it->first ;
            std::vector<int> &indices = it->second ;
            T* boxes_ = h_decodedBoxes + boxIdxOffset;
            T* confPreds_ = h_permutedConfPreds + confIdxOffset + label * numPriors ;

            int numIndices = indices.size() ;
            for (int j = 0 ; j < numIndices ; ++j) {
                int idx = indices[j] ;
                output[outHeight * i * 6 + count ] = label + 1 ; // MATLAB +1
                output[outHeight * i * 6 + outHeight + 1 * count] = confPreds_[idx] ;
                output[outHeight * i * 6 + outHeight * 2 + count] = boxes_[idx * 4] ;
                output[outHeight * i * 6 + outHeight * 3 + count] = boxes_[idx * 4 + 1] ;
                output[outHeight * i * 6 + outHeight * 4 + count] = boxes_[idx * 4 + 2] ;
                output[outHeight * i * 6 + outHeight * 5 + count] = boxes_[idx * 4 + 3] ;
              ++count;
            }
        }
    }
    delete[] h_decodedBoxes ;
    delete[] h_permutedConfPreds ;

    return VLE_Success ;
   }
 } ;
} } // namespace vl::impl

template struct vl::impl::multiboxdetector<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::multiboxdetector<vl::VLDT_GPU, double> ;
#endif
