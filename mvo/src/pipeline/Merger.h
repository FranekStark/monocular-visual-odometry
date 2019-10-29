//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_MERGER_H_
#define MVO_SRC_MERGER_H_

#include "PipelineStage.h"
#include "../operations/FeatureOperations.h"
#include "../operations/VisualisationUtils.hpp"

class Merger : public PipelineStage {
 private:
  Frame* _preFrame;
  Frame* _keepFrame;
  ros::Time _lastFrameTime;

 public:
  Merger(PipelineStage &precursor,
         unsigned int outGoingChannelSize);
  ~Merger() override;
  Frame* stage(Frame * newFrame) override;
};

#endif //MVO_SRC_MERGER_H_
