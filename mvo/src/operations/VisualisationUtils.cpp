//
// Created by franek on 26.09.19.
//

#include "VisualisationUtils.hpp"
#include "../sliding_window/SlidingWindow.hpp"

void VisualisationUtils::drawFeatures(const Frame &frame, cv::Mat &image) {
  frame.lock();
  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    cv::Scalar color;
    if (feature->_preFeature < 0) { //Blue, if ne prefeatures
      color = cv::Scalar(255, 0, 0); //BGR
    } else { //If tracked, than GREEN
      color = cv::Scalar(0, 255, 0);
    }
    cv::circle(image, cv::Point(feature->_positionImage), 10, color, -10);
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
  auto baseLine = SlidingWindow::getBaseLineToPrevious(frame);
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