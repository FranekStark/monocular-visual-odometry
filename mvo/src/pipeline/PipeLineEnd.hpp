//
// Created by franek on 26.09.19.
//

#ifndef MVO_SRC_PIPELINE_PIPELINEEND_HPP_
#define MVO_SRC_PIPELINE_PIPELINEEND_HPP_
#include "PipelineStage.h"

class PipeLineEnd : public PipelineStage{
 private:
  Frame * _lastFrame;
  Frame *stage(Frame *newFrame) override;
 public:
  ~PipeLineEnd() override = default;
  PipeLineEnd(PipelineStage &precursor);

};

#endif //MVO_SRC_PIPELINE_PIPELINEEND_HPP_