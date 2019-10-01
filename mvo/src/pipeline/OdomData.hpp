//
// Created by franek on 27.09.19.
//

#ifndef MVO_SRC_PIPELINE_ODOMDATA_HPP_
#define MVO_SRC_PIPELINE_ODOMDATA_HPP_

#include <opencv2/core.hpp>

struct OdomData {
  cv::Point3d position;
  cv::Matx33d orientation;
};

#endif //MVO_SRC_PIPELINE_ODOMDATA_HPP_
