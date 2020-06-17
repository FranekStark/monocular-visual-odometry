//
// Created by franek on 26.09.19.
//

#ifndef MVO_SRC_PIPELINE_PIPELINEEND_HPP_
#define MVO_SRC_PIPELINE_PIPELINEEND_HPP_
#include "PipelineStage.h"

class PipeLineEnd : public PipelineStage{
 private:
  std::function<void(std::vector<cv::Vec3d>&, cv::Point3d, cv::Scalar)> _drawProjectionsCallback;
  Frame * _lastFrame;
  Frame *stage(Frame *newFrame) override;
#ifdef RATINGDATA
  static void printRatingData(const Frame *frame);
#endif

 public:
  ~PipeLineEnd() override = default;
  PipeLineEnd(PipelineStage &precursor,
              std::function<void(std::vector<cv::Vec3d>&, cv::Point3d, cv::Scalar)> drawProjectionsCallback
      );

};

#endif //MVO_SRC_PIPELINE_PIPELINEEND_HPP_
