#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <image_geometry/pinhole_camera_model.h>
#include "Feature.hpp"
#include <mutex>
#include "../Utils.hpp"

enum FrameType {
  TEMP, PERSIST
};

struct Frame {
  std::vector<Feature> _features;
  std::vector<cv::Mat> _imagePyramide;
  image_geometry::PinholeCameraModel _cameraModel;

  cv::Vec3d _baseLine;
  cv::Matx33d _rotation;

  Frame *_preFrame;
  mutable std::mutex _lock;

  void lock() const;
  void unlock() const;
};

#endif //FRAME_HPP