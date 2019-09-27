//
// Created by franek on 25.09.19.
//

#ifndef MVO_SRC_BASELINEESTIMATOR_HPP_
#define MVO_SRC_BASELINEESTIMATOR_HPP_

#include "PipelineStage.h"
#include "../algorithms/EpipolarGeometry.hpp"
#include "../operations/FeatureOperations.h"
#include "OdomData.hpp"

class BaselineEstimator: public PipelineStage {
 private:
  Frame * _prevFrame;
  EpipolarGeometry & _epipolarGeometry;
 public:
  BaselineEstimator(PipelineStage &precursor,
                    unsigned int out_going_channel_size,
                    EpipolarGeometry &epipolarGeometry);
  ~BaselineEstimator() override ;
  Frame *stage(Frame *newFrame) override;
  Channel<OdomData> _baseLine;

};

#endif //MVO_SRC_BASELINEESTIMATOR_HPP_
