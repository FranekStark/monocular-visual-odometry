#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <opencv2/core.hpp>

struct Feature {
  cv::Point2f _positionImage;
  cv::Vec3d _positionEuclidian;

  /*Points to feature Before, if no feature Before exists -1*/
  int _preFeature;
  /*Number of Features Before*/
  unsigned int _preFeatureCounter;

  Feature(cv::Point2f positionImage, cv::Vec3d positionEucilidan, int preFeature, unsigned int preFeatureCounter)
      : _positionImage(positionImage),
        _positionEuclidian(positionEucilidan),
        _preFeature(preFeature),
        _preFeatureCounter(preFeatureCounter) {
  }
};

#endif  // FEATRUE_HPP