//
// Created by franek on 25.09.19.
//

#ifndef MVO_SRC_BASELINEESTIMATOR_HPP_
#define MVO_SRC_BASELINEESTIMATOR_HPP_

#include "PipelineStage.h"
#include "EpipolarGeometry.hpp"
#include "FeatureOperations.h"

class BaselineEstimator: PipelineStage {
 private:
  Frame * _prevFrame;
  EpipolarGeometry & _epipolarGeometry;
 public:
  BaselineEstimator(SlidingWindow &sliding_window,
                    PipelineStage &precursor,
                    unsigned int out_going_channel_size,
                    EpipolarGeometry &epipolarGeometry);
  ~BaselineEstimator();

 private:
  Frame *stage(Frame *newFrame) override;

};

#endif //MVO_SRC_BASELINEESTIMATOR_HPP_
