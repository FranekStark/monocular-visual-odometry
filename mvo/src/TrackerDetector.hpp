//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_TRACKERDETECTOR_HPP_
#define MVO_SRC_TRACKERDETECTOR_HPP_

#include "PipelineStage.h"
#include "sliding_window/SlidingWindow.hpp"
#include "FeatureOperations.h"
#include "mvo.hpp"

class TrackerDetector : PipelineStage{
 private:
  /**
   * Reference to the previous Frame. Has already been passed to successor.
   */
  Frame * _prevFrame;
  CornerTracking & _cornerTracking;
  unsigned int _numberToDetect;

  void track(Frame & newFrame);
  void detect(Frame & newFrame, unsigned int number);

 public:
  TrackerDetector(SlidingWindow &slidingWindow,
                  PipelineStage &precursor,
                  unsigned int outGoingChannelSize,
                  CornerTracking &cornerTracking,
                  unsigned int number);
  ~TrackerDetector();
  Frame* stage(Frame * newFrame) override;
};

#endif //MVO_SRC_TRACKERDETECTOR_HPP_
