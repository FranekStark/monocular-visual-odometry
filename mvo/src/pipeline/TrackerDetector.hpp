//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_TRACKERDETECTOR_HPP_
#define MVO_SRC_TRACKERDETECTOR_HPP_

#include <opencv2/highgui.hpp>

#include "PipelineStage.h"
#include "../sliding_window/Frame.hpp"
#include "../operations/FeatureOperations.h"
#include "../operations/VisualisationUtils.hpp"
#include "../algorithms/CornerTracking.hpp"

class TrackerDetector : public PipelineStage{
 private:
  /**
   * Reference to the previous Frame. Has already been passed to successor.
   */
  Frame * _prevFrame;
  CornerTracking & _cornerTracking;

  void track(Frame & newFrame);
  void detect(Frame & newFrame, unsigned int number);

 public:
  TrackerDetector(PipelineStage &precursor,
                  unsigned int outGoingChannelSize,
                  CornerTracking &cornerTracking);
  ~TrackerDetector() override;
  Frame* stage(Frame * newFrame) override;
  static cv::Rect2d getShipMask(const cv::Size &imageSize, double shipHeight, double shipWidth);
};

#endif //MVO_SRC_TRACKERDETECTOR_HPP_
