//
// Created by franek on 24.09.19.
//

#ifndef MVO_SRC_SLIDING_WINDOW_FEATUREOPERATIONS_H_
#define MVO_SRC_SLIDING_WINDOW_FEATUREOPERATIONS_H_

#include <opencv2/core.hpp>
#include <image_geometry/pinhole_camera_model.h>

class FeatureOperations {
 public:
  static double calcDisparity(const std::vector<cv::Vec3d> &first, const std::vector<cv::Vec3d> &second);
  /**
   * Project camera-pixel-coordinate Feature into camera-projected Feature
   * @param features source vector
   * @param featuresE empty target vector
   * @param cameraModel parameters of the camera
   */
  static void euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> &featuresE,
                                 const image_geometry::PinholeCameraModel &cameraModel);

  /**
   * Unproject camera-projected Features into pixel-coordinates
   * @param featuresE source vector
   * @param features emtpy target vector
   * @param cameraModel  parameters of the camera
   */
  static void euclidUnNormFeatures(const std::vector<cv::Vec3d> &featuresE, std::vector<cv::Point2f> &features,
                                   const image_geometry::PinholeCameraModel &cameraModel);

  /**
   * Unrotates features from source- into target-vector
   * @param features source vector
   * @param unrotatedFeatures empty target vector
   * @param R the rotation between the features
   */
  static void unrotateFeatures(const std::vector<cv::Vec3d> &features, std::vector<cv::Vec3d> &unrotatedFeatures,
                               const cv::Matx33d &R);

  /**
   * Reconstructs the Depth of two detections of the same Feature on two different places (triangulation)
   * @param depth the target vector (first empty)
   * @param m2L second detections of the points
   * @param m1L first detections of the point
   * @param r the rotation between first and second
   * @param b the baseline between first and second
   */
  static void reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                               const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r,
                               const cv::Vec3d &b);


};

#endif //MVO_SRC_SLIDING_WINDOW_FEATUREOPERATIONS_H_
