//
// Created by franek on 24.09.19.
//

#include "FeatureOperations.h"

double FeatureOperations::calcDisparity(const std::vector<cv::Vec3d> &first, const std::vector<cv::Vec3d> &second) {
  assert(first.size() == second.size());
  if (first.size() == 0) {
    return 0;
  }
  //
  double diff = 0;
  for (auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++) {
    diff +=  (1 - ((p2->dot(*p1))/(cv::norm(*p2)*cv::norm(*p1)))); //TODO: What about negativ Scalar products?
  }
  diff = diff / first.size();
  //
  return diff;
}


void FeatureOperations::euclidNormFeatures(const std::vector<cv::Point2f> &features,
                                           std::vector<cv::Vec3d> &featuresE,
                                           const image_geometry::PinholeCameraModel &cameraModel) {
  //Check size of target vector
  assert(featuresE.size() == 0);
  //Reserve memory
  featuresE.reserve(features.size());
  //Project Feature and push_back
  for (auto feature = features.begin(); feature != features.end(); feature++) {
    cv::Vec3d projected = cameraModel.projectPixelTo3dRay(*feature);
    //cv::Vec3d projected_normed = projected/cv::norm(projected);
    featuresE.push_back(projected);
  }
}

void FeatureOperations::euclidUnNormFeatures(const std::vector<cv::Vec3d> &featuresE,
                                             std::vector<cv::Point2f> &features,
                                             const image_geometry::PinholeCameraModel &cameraModel) {
  //Check size of target vector
  assert(features.empty());
  //Reserve memory
  features.reserve(featuresE.size());
  //Unproject Feature and push_back
  for (auto featureE = featuresE.begin(); featureE != featuresE.end(); featureE++) {
    features.push_back(cameraModel.project3dToPixel(*featureE));
  }
}

void FeatureOperations::unrotateFeatures(const std::vector<cv::Vec3d> &features,
                                         std::vector<cv::Vec3d> &unrotatedFeatures,
                                         const cv::Matx33d &R) {
  //Check size of target vector
  assert(unrotatedFeatures.empty());
  //Save Memory on target vector
  unrotatedFeatures.reserve(features.size());
  //Unrotate Feature and push_back into target
  for (auto feature = features.begin(); feature != features.end(); feature++) {
    auto unrotatedFeature = R * (*feature);
    unrotatedFeature = unrotatedFeature / unrotatedFeature(2);
    unrotatedFeatures.push_back(unrotatedFeature);
  }
}

void FeatureOperations::reconstructDepth(std::vector<double> &depth,
                                         const std::vector<cv::Vec3d> &m2L,
                                         const std::vector<cv::Vec3d> &m1L,
                                         const cv::Matx33d &r,
                                         const cv::Vec3d &b) {
  //First check tge Vector sizes
  assert(depth.size() == 0);
  assert(m1L.size() == m2L.size());
  //Reserve memory
  depth.reserve(m1L.size());
  //TODO: What the fuck is happening here?
  for (auto m1 = m1L.begin(), m2 = m2L.begin(); m1 != m1L.end() && m2 != m2L.end(); m1++, m2++) {
    cv::Matx33d C;
    C << 1, 0, -(*m2)(0),  //
        0, 1, -(*m2)(1),   //
        -(*m2)(0), -(*m2)(1), (*m2)(0) * (*m2)(0) + (*m2)(1) * (*m2)(1);
    double Z = (m1->t() * r * C * r.t() * b)(0) / (m1->t() * r * C * r.t() * (*m1))(0);
    depth.push_back(Z);
  }
}

void FeatureOperations::normFeatures(std::vector<cv::Vec3d> &featuresE) {
  for(auto feature = featuresE.begin(); feature != featuresE.end(); feature++){
    double scale = 1.0/cv::norm(*feature);
    *feature = *feature * scale;
  }
}
