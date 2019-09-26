//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_MERGER_H_
#define MVO_SRC_MERGER_H_

#include "PipelineStage.h"
#include "../operations/FeatureOperations.h"
#include "../operations/VisualisationUtils.hpp"
#include "../mvo.hpp"

class Merger : PipelineStage {
 private:
  Frame* _preFrame;
  Frame* _keepFrame;

  double _sameDisparityThreshold;
  double _movementDisparityThreshold;

 public:
  Merger(PipelineStage &precursor,
         unsigned int outGoingChannelSize,
         double sameThreshold,
         double movementThreshold);
  ~Merger();
  Frame* stage(Frame * newFrame) override;
};

#endif //MVO_SRC_MERGER_H_
