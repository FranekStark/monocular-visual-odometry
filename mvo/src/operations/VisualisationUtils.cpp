//
// Created by franek on 26.09.19.
//

#include "VisualisationUtils.hpp"

void VisualisationUtils::drawFeatures(const Frame &frame, cv::Mat &image) {
  frame.lock();
  for (const auto & _feature : frame._features) {
    cv::Scalar color;
    if (_feature._preFeature < 0) { //Blue, if ne prefeatures
      color = cv::Scalar(255, 0, 0); //BGR
    } else { //If tracked, than GREEN
      color = cv::Scalar(0, 255, 0);
    }
    cv::circle(image, cv::Point(_feature._positionImage), 10, color, -10);
  }
  frame.unlock();
}

void VisualisationUtils::drawCorrespondences(const Frame &oldestFrame, const Frame &newestFrame, cv::Mat &image) {
  //Lock newest Frame
  newestFrame.lock();
  //Check Depth
  unsigned int depth = 0;
  {
    const Frame *frame = &newestFrame;
    while (frame != &oldestFrame) {
      frame = frame->_preFrame;
      //Lock Prefame;
      frame->lock();
      depth++;
    }
  }

  for (auto feature = newestFrame._features.begin(); feature != newestFrame._features.end(); feature++) {
    if (feature->_preFeatureCounter >= depth) {
      const Frame *frame = &newestFrame;
      unsigned int counter = 0;
      cv::circle(image, feature->_positionImage, 5, cv::Scalar(0, 0, 255), -1);
      cv::putText(image,
                  std::to_string(counter),
                  feature->_positionImage,
                  cv::FONT_HERSHEY_PLAIN,
                  5,
                  cv::Scalar(255, 255, 0));
      while (frame != &oldestFrame) {
        counter++;
        //Line to next
        cv::line(image,
                 feature->_positionImage,
                 frame->_preFrame->_features[feature->_preFeature]._positionImage,
                 cv::Scalar(255, 255, 255),
                 4);
        //Next
        cv::circle(image, feature->_positionImage, 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(image,
                    std::to_string(counter),
                    frame->_preFrame->_features[feature->_preFeature]._positionImage,
                    cv::FONT_HERSHEY_PLAIN,
                    5,
                    cv::Scalar(255, 255, 0));
        frame = frame->_preFrame;
      }
    }
  }

  //Unlock:
  //Lock newest Frame
  newestFrame.unlock();
  //Check Depth
  {
    const Frame *frame = &newestFrame;
    while (frame != &oldestFrame) {
      frame = frame->_preFrame;
      //Lock Prefame;
      frame->unlock();
    }
  }

}

void VisualisationUtils::drawMovementDebug(const Frame &frame,
                                           const cv::Scalar &color,
                                           cv::Mat &image,
                                           unsigned int index) {
  frame.lock();
  auto baseLine = frame._baseLine;
  auto rotation = frame._rotation;
  //BaseLine into Frame-CameraCorrdinates
  baseLine = rotation.t() * baseLine;
  double norm = cv::norm(baseLine);
  auto baseLineNorm = cv::normalize(baseLine);
  frame.unlock();

  int mitX = double(image.cols) / 2.0;
  int mitY = double(image.rows) / 2.0;
  double scaleX = (image.cols - mitX) / 1.5;
  double scaleY = (image.rows - mitY) / 1.5;
  cv::arrowedLine(image, cv::Point(mitX, mitY),
                  cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY), color, 10);
  cv::line(image, cv::Point(mitY, index * 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), index * 20), color, 10);
  double scaleMin = std::min(scaleX, scaleY);
  cv::circle(image, cv::Point2d(mitX, mitY), scaleMin * norm, color, 10);
}

void VisualisationUtils::drawFeaturesUnrotated(cv::Mat &image,
                                               const std::vector<cv::Point2f> &featuresBefore,
                                               const std::vector<cv::Point2f> &featuresNew,
                                               const std::vector<cv::Point2f> &featuresNewUnrotated) {
  assert(featuresBefore.size() == featuresNew.size());
  assert(featuresNew.size() == featuresNewUnrotated.size());
  auto featureBefore = featuresBefore.begin();
  auto featureNew = featuresNew.begin();
  auto featureNewUnrotaed = featuresNewUnrotated.begin();
  while (featureBefore != featuresBefore.end()) {
    cv::circle(image, cv::Point(*featureBefore), 10, cv::Scalar(255, 0, 0), -10);
    cv::circle(image, cv::Point(*featureNew), 10, cv::Scalar(0, 255, 0), -10);
    cv::circle(image, cv::Point(*featureNewUnrotaed), 10, cv::Scalar(0, 0, 255), -10);
    featureBefore++;
    featureNew++;
    featureNewUnrotaed++;
  }
}
void VisualisationUtils::drawCorrespondences(const std::vector<std::vector<cv::Vec3d> *> vectors,
                                             const image_geometry::PinholeCameraModel &camera,
                                             cv::Mat &image) {
  for(unsigned int featureIndex = 0;  featureIndex < vectors[0]->size(); featureIndex++){
    for(unsigned int frameIndex = 0; frameIndex < vectors.size(); frameIndex++){
      //Point
      auto & point3d = (*vectors[frameIndex])[featureIndex];
      cv::Point2f point2f = camera.project3dToPixel(point3d);
      cv::circle(image, point2f, 5, cv::Scalar(0, 0, 255), -1);
      cv::putText(image,
                  std::to_string(frameIndex),
                  point2f,
                  cv::FONT_HERSHEY_PLAIN,
                  5,
                  cv::Scalar(255, 255, 0));

      if(frameIndex != (vectors.size() -1)){//If not Last one
        //Line
        auto & point3dNEXT =  (*vectors[frameIndex + 1])[featureIndex];
        cv::Point2f point2fNEXT = camera.project3dToPixel(point3dNEXT);
        cv::line(image,
                 point2f,
                 point2fNEXT,
                 cv::Scalar(255, 255, 255),
                 4);
      }
    }
  }

}
void VisualisationUtils::drawRect(cv::Mat &image, const cv::Rect &rect) {
  cv::Mat maskImage = image.clone();
  cv::rectangle(maskImage, rect, cv::Scalar(0, 0, 255), -1);
  cv::addWeighted(image, 0.5, maskImage, 0.5, 0.0, image);
}
