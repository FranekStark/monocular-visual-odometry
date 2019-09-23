#include <opencv2/core.hpp>

#ifndef ODOMDATA_HPP
#define ODOMDATA_HPP

struct OdomData {
  cv::Vec3d b;
  cv::Vec3d s;
  cv::Matx33d o;
};

#endif  //ODOMDATA_HPP