//
// Created by franek on 26.09.19.
//

#include "PipelineBegin.hpp"
Frame *PipelineBegin::stage(Frame *newFrame) {
  ROS_ERROR("Pipeline Begin is not itself a Thread!");
  return nullptr;
}

PipelineBegin::PipelineBegin() : PipelineStage(nullptr,
                                               1) {}
PipelineBegin::~PipelineBegin() {
}

void PipelineBegin::pipeIn(Frame *frame) {
  _outGoingChannel.enqueue(frame);
}
