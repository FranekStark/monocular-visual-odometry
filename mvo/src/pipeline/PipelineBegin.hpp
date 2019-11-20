//
// Created by franek on 26.09.19.
//

#ifndef MVO_SRC_PIPELINE_PIPELINEBEGIN_HPP_
#define MVO_SRC_PIPELINE_PIPELINEBEGIN_HPP_

#include "PipelineStage.h"

class PipelineBegin : public PipelineStage{
 public:
  PipelineBegin();
  ~PipelineBegin() override = default;
 protected:
  void pipeIn(Frame * frame);
 private:
  Frame *stage(Frame *newFrame) override;

};

#endif //MVO_SRC_PIPELINE_PIPELINEBEGIN_HPP_
