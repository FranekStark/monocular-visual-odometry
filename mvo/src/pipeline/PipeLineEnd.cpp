//
// Created by franek on 26.09.19.
//

#include "PipeLineEnd.hpp"
PipeLineEnd::PipeLineEnd(PipelineStage &precursor) : PipelineStage(&precursor,
                                                                   1),
                                                     _lastFrame(nullptr) {}
Frame *PipeLineEnd::stage(Frame *newFrame) {
  if (_lastFrame != nullptr) { //Only if there is a Last Frame
    LOG_DEBUG("Deleted end Frame: " << _lastFrame);
    //Delete kept Frame
    delete _lastFrame;
  }
  //Keep the new One, cause the preStages could need the Data
  _lastFrame = newFrame;
  return nullptr;
}
