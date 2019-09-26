//
// Created by franek on 26.09.19.
//

#include "VisualisationUtils.hpp"

void VisualisationUtils::drawFeatures(const Frame &frame, cv::Mat &image) {
  frame._lock.lock();
  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    cv::Scalar color;
    if (feature._preFeature < 0) { //Blue, if ne prefeatures
      color = cv::Scalar(255, 0, 0); //BGR
    } else { //If tracked, than GREEN
      color = cv::Scalar(0, 255, 0);
    }
    cv::circle(image, cv::Point(feature._positionImage), 10, color, -10);
  }
  frame._lock.unlock();
}

void VisualisationUtils::drawCorrespondences(const Frame &oldestFrame, const Frame &newestFrame, cv::Mat &image) {
  //Lock newest Frame
  newestFrame._lock.lock();
  //Check Depth
  unsigned int depth = 0;
  {
    Frame *frame = &newestFrame;
    while (frame != &oldestFrame) {
      frame = frame->_preFrame;
      //Lock Prefame;
      frame->_lock.lock();
      depth++
    }
  }

  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    if (feature->_preFeatureCounter >= depth) {
      Frame *frame = &newestFrame;
      Feature *feature = feature;
      unsigned  int counter = 0;
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
                 frame->_preFrame._features[feature->_preFeature]._positionImage,
                 cv::Scalar(255, 255, 255),
                 4);
        //Next
        cv::circle(image, feature->_positionImage, 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(image,
                    std::to_string(counter),
                    frame->_preFrame._features[feature->_preFeature]._positionImage,
                    cv::FONT_HERSHEY_PLAIN,
                    5,
                    cv::Scalar(255, 255, 0));
        frame = frame->_preFrame;
      }
    }
  }

  //Unlock:
  //Lock newest Frame
  newestFrame._lock.unlock();
  //Check Depth
  {
    Frame *frame = &newestFrame;
    while (frame != &oldestFrame) {
      frame = frame->_preFrame;
      //Lock Prefame;
      frame->_lock.unlock();
    }
  }

}

void VisualisationUtils::drawMovementDebug(const Frame &frame, const cv::Scalar &color, cv::Mat &image, unsigned int index) {
  frame._lock.lock();
  double norm = cv::norm(baseLine);
  auto baseLineNorm = cv::normalize(baseLine);
  frame._lock.unlock()

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

void VisualisationUtils::drawFeaturesUnrotated(const Frame &frame, cv::Mat &image) {
  frame._lock.lock();
  assert(frame._preFrame != nullptr);
  auto rDiff = frame._preFrame->_rotation.t() * frame._rotation;
  for(auto feature = frame._features.begin(); feature != frame._features.end(); feature++){
    if(feature->_preFeature >= 0){ //Only if it has a Prefeature
      //Prefeature
      auto &  = frame._preFrame._features[feature._preFeature]._positionImage;
      //Feature normal
      auto & feature = feature._positionImage;
      //Feature unrotated
      auto & featureE = feature._positionEuclidian;
      auto featureEUnrotated = rDiff * featureE;
      auto featureUnrotated = frame._cameraModel.project3dToPixel(featureEUnrotated);
      //Draw
      cv::circle(image, cv::Point(preFeature), 10, cv::Scalar(255,0,0), -10);
      cv::circle(image, cv::Point(feature), 10, cv::Scalar(0,255,0), -10);
      cv::circle(image, cv::Point(featureUnrotated), 10, cv::Scalar(0,0,255), -10);
    }
  }
  frame._lock.unlock();
}