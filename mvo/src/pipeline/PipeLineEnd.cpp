//
// Created by franek on 26.09.19.
//

#include "PipeLineEnd.hpp"
PipeLineEnd::PipeLineEnd(PipelineStage *precursor) : PipelineStage(precursor,
                                                                   1) {}
Frame *PipeLineEnd::stage(Frame *newFrame) {
  //Delete kept Frame
  delete _lastFrame;
  //Keep the new One, cause the preStages could need the Data
  _lastFrame = newFrame;
  return nullptr;
}

PipeLineEnd::~PipeLineEnd() {

}
