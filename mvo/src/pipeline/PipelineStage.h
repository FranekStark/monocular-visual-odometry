//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_PIPELINESTAGE_H_
#define MVO_SRC_PIPELINESTAGE_H_


#include "../pareigis_lib/Channel.hpp"
#include <ros/ros.h>
#include "../Utils.hpp"
#include "../sliding_window/Frame.hpp"

class PipelineStage {
 protected:
  PipelineStage * _precursorStage;
  Channel<Frame *> _outGoingChannel;

 public:
  PipelineStage(
      PipelineStage *precursor,
      unsigned int outGoingChannelSize);
  virtual ~PipelineStage();
  void operator()();

  /**
   * This abstract Function is the main "Pipe-Line-Logic-Function".
   * If a new Frame occurs, than it will be called.
   * Tha it has two possiblities. Return a Frame, which will be passed to the following
   * Pipelinestage or rreturn nullptr, to indicate, that in this Pipelinestep no outgoing
   * Frame is ready.
   *
   * @param newFrame the incoming new Frame
   * @return the ougoing Frame or nllptr
   */
  virtual Frame* stage(Frame * newFrame) = 0;
};

#endif //MVO_SRC_PIPELINESTAGE_H_
