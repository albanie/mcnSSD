// @file multiboxdetector.hpp
// @brief Multibox Detector
// @author Samuel Albanie
// @author Andrea Vedaldi

/*
Copyright (C) 2017- Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_MULTIBOXDETECTOR_H
#define VL_MULTIBOXDETECTOR_H

#include <bits/data.hpp>
#include <cstddef>

// defines the dispatcher for CUDA kernels:
namespace vl { namespace impl {

  template<vl::DeviceType dev, typename T>
  struct multiboxdetector {

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
            size_t numPriors) ;
  } ;

} }

#endif /* defined(VL_MULTIBOXDETECTOR_H) */
