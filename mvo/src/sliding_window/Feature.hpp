#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <opencv2/core.hpp>

struct Feature {
  cv::Point2f _positionImage;
  cv::Vec3d _positionProjected;

  /*Points to feature Before, if no feature Before exists -1*/
  int _preFeature;
  /*Number of Features Before*/
  unsigned int _preFeatureCounter;

  Feature(cv::Point2f positionImage, cv::Vec3d positionProjected, int preFeature, unsigned int preFeatureCounter)
      : _positionImage(positionImage),
        _positionProjected(positionProjected),
        _preFeature(preFeature),
        _preFeatureCounter(preFeatureCounter) {
  }
};

#endif  // FEATRUE_HPP