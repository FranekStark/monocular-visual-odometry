#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <image_geometry/pinhole_camera_model.h>
#include "Feature.hpp"
#include <mutex>

enum FrameType {
  TEMP, PERSIST
};

struct Frame {
  std::vector<Feature> _features;
  cv::Mat _image;
  image_geometry::PinholeCameraModel _cameraModel;

  cv::Vec3d _baseLine;
  cv::Matx33d _rotation;

  Frame *_preFrame;
  mutable std::mutex _lock;
};

#endif //FRAME_HPP