//
// Created by franek on 24.09.19.
//

#include "PipelineStage.h"

PipelineStage::PipelineStage(PipelineStage *precursor,
                             unsigned int outGoingChannelSize) :
    _precursorStage(precursor),
    _outGoingChannel(outGoingChannelSize) {
}

PipelineStage::~PipelineStage() {
  _outGoingChannel.destroy();
}

void PipelineStage::operator()() {
  while (ros::ok()) {
    /*Block on new Frame*/
    Frame *ingoingFrame = _precursorStage->_outGoingChannel.dequeue();
    ROS_INFO_STREAM( Utils::GetThreadName() << " ingoing Queue: " <<_precursorStage->_outGoingChannel.size());
    LOG_DEBUG("New Frame in PipelineStage: " << ingoingFrame);
    /*Process the new Frame and get an Outgoing Frame*/
    Frame *outgoingFrame = stage(ingoingFrame);
    /*If the Outgoing Frame is not null than pass it on*/
    if (outgoingFrame != nullptr) {
      _outGoingChannel.enqueue(outgoingFrame);
      ROS_INFO_STREAM( Utils::GetThreadName() << " outgoing Queue: " <<_outGoingChannel.size());
      LOG_DEBUG("Piped Through: " << outgoingFrame);
    }
  }
}