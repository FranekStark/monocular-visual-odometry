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
                                         const std::vector<cv::Vec3d> &vec1,
                                         const std::vector<cv::Vec3d> &vec0,
                                         const cv::Matx33d &R1,
                                         const cv::Matx33d &R0,
                                         const cv::Vec3d &b) {
  //First check tge Vector sizes
  assert(depth.size() == 0);
  assert(vec1.size() == vec0.size());
  //Reserve memory
  depth.reserve(vec0.size());
  //TODO: What the fuck is happening here?
  for (auto m0 = vec0.begin(), m1 = vec1.begin(); m1 != vec1.end() && m0 != vec0.end(); m1++, m0++) {
    //Calculate the Angles between the Featureprojections and the Baseline
    double  anglem1 = acos((R1 * *m1).dot(b) / (cv::norm(R1 * *m1) * cv::norm(b)));
    double  anglem0 = acos((R0 * *m0).dot(-b) / (cv::norm(R1 * *m0) * cv::norm(b)));

    double anglesum = anglem0 + anglem1;
    if(anglesum > M_PI){ //Not Bigger than 180 Degrees!
      depth.push_back(-1.0);
    }else{
      depth.push_back(1.0);
    }
  }
}

void FeatureOperations::normFeatures(std::vector<cv::Vec3d> &featuresE) {
  for(auto feature = featuresE.begin(); feature != featuresE.end(); feature++){
    double scale = 1.0/cv::norm(*feature);
    *feature = *feature * scale;
  }
}
