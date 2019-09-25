//
// Created by franek on 24.09.19.
//

#include "PipelineStage.h"

PipelineStage::PipelineStage(SlidingWindow &slidingWindow,
                             PipelineStage &precursor,
                             unsigned int outGoingChannelSize) :
_slidingWindow(slidingWindow),
_precursorStage(precursor),
_outGoingChannel(outGoingChannelSize)
{
}

PipelineStage::~PipelineStage() {
  _outGoingChannel.destroy();
};

void PipelineStage::operator()() {
  /*Block on new Frame*/
  Frame* ingoingFrame = _precursorStage._outGoingChannel.dequeue();
  /*Process the new Frame and get an Outgoing Frame*/
  Frame* outgoingFrame = stage(ingoingFrame);
  /*If the Outgoing Frame is not null than pass it on*/
  if(outgoingFrame != null){
    _outGoingChannel.enqueue(outgoingFrame);
  }
}