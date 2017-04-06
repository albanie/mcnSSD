// @file nnmultiboxdetector.hpp
// @brief Multibox Detector block
// @author Samuel Albanie 
// @author Andrea Vedaldi
/*
Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnmultiboxdetector__
#define __vl__nnmultiboxdetector__

#include <bits/data.hpp>
#include <stdio.h>

namespace vl {

  vl::ErrorCode
  nnmultiboxdetector_forward(vl::Context& context,
                             vl::Tensor output,
                             vl::Tensor locPreds,
                             vl::Tensor confPreds,
                             vl::Tensor priors,
                             int nmsTopK,
                             int keepTopK,
                             int numClasses,
                             float nmsThresh,
                             float confThresh,
                             int backgroundLabel) ;
}

#endif /* defined(__vl__nnmultiboxdetector__) */
