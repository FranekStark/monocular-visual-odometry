//
// Created by franek on 15.11.19.
//

#ifndef MVO_SRC_PIPELINE_SCALER_HPP_
#define MVO_SRC_PIPELINE_SCALER_HPP_

#include <vector>
#include <ros/ros.h>

#include "PipelineStage.h"
class Scaler : public PipelineStage{
 public:
  double _highestScale = 0;
  double _lowestScale = 0;
  Scaler(PipelineStage &precursor, unsigned int out_going_channel_size);
  Frame *stage(Frame *newFrame) override;
  virtual ~Scaler();
 private:
  Frame * _prevFrame;
  Frame * _keptFrame;

};

#endif //MVO_SRC_PIPELINE_SCALER_HPP_
