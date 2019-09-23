#include "CornerTracker.hpp"
#include  <ros/ros.h>

CornerTracker::CornerTracker() : _blockSize(3), _minDifPercent(0.02), _qualityLevel(0.4), _windowSize(21, 21),
                                 _maxPyramideLevel(3) {
}

CornerTracker::~CornerTracker() {
}

void CornerTracker::detectFeatures(std::vector<cv::Point2f> &corner, const cv::Mat &image, int numberToDetect,
                                   const std::vector<cv::Point2f> &existingFeatures, cv::Rect2d &mask,
                                   bool forceDetection) {
  if (numberToDetect <= 0) {
    return;
  }
  //Create Mask
  cv::Mat maskImage(image.size(), CV_8U);
  maskImage = cv::Scalar(255); //First use all Pixels
  double imageDiag = sqrt(image.rows * image.rows + image.cols * image.cols);
  double mindistance = _minDifPercent * imageDiag;
  for (auto existingFeature : existingFeatures) {
    cv::circle(maskImage, existingFeature, mindistance, cv::Scalar(0),
               -1); //Here no Feature Detection! //TODO: param
  }
  cv::rectangle(maskImage, mask, cv::Scalar(0), cv::FILLED);

  double qualityLevel = _qualityLevel;
  if (forceDetection) {
    qualityLevel = 0.01; //Set Quality Low, to Detect as much features as possible
  }
  cv::goodFeaturesToTrack(image, corner, numberToDetect, qualityLevel, mindistance, maskImage, _blockSize,
                          bool(true),
                          _k);  // Corners berechnen TODO: More params -> especially the detection Quality

  // Subpixel-genau:
  if (corner.size() > 0) {
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria =
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(image, corner, winSize, zeroZone, criteria);
  }
}

void CornerTracker::trackFeatures(const cv::Mat &currentPyramide,
                                  const std::vector<cv::Point2f> &prevFeatures,
                                  std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found,
                                  cv::Rect2d &mask) {
  /**
   * Params
   **/
  cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
  int flags = 0;

  /**
   * If there a Features in Vector 'tracked Features', we use them as inital Estimation.
   **/
  if (prevFeatures.size() == trackedFeatures.size()) {
    flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  }


  // TODO: Handle Case, when no before Features!
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(_beforePyramide, currentPyramide, prevFeatures, trackedFeatures, found, error, _windowSize,
                           _maxPyramideLevel, criteria, flags);  // TODO: more
  _beforePyramide = currentPyramide;


  double imageDiag = sqrt(currentPyramide[0].rows * currentPyramide[0].rows + currentPyramide[0].cols * currentPyramide[0].cols);
  double mindistance = _minDifPercent * imageDiag;

  //TODO: Sort out to Close features!
  std::vector<cv::Point2f>::iterator f1, f2;
  std::vector<unsigned char>::iterator f1Found, f2Found;
  for (f1 = trackedFeatures.begin(), f1Found = found.begin(); f1 != trackedFeatures.end(); f1++, f1Found++) {
    if (*f1Found == 0) {
      continue;
    }

    //Features in Mask:
    if (mask.contains(*f1)) {
      *f1Found = 0;
      continue;
    }

    for (f2 = trackedFeatures.begin(), f2Found = found.begin(); f2 != trackedFeatures.end(); f2++, f2Found++) {
      if (*f2Found == 0) {
        continue;
      }
      if (f1 == f2) {
        continue; //Don't sort out the same Feature
      }

      if (cv::norm(*f1 - *f2) < mindistance) { //TODO: Param
        *f2Found = 0; //Sort Out
      }

    }
  }

}

cv::Mat CornerTracker::createPyramide(cv::Mat image) const {
  cv::Mat result;
  cv::buildOpticalFlowPyramid(image, result, _windowSize, _maxPyramideLevel);
  return result;
}

void CornerTracker::setCornerDetectorParams(int blockSize, double minDifPercent, double qualityLevel, int windowSize,
                                            int maxPyramideLevel) {
  _blockSize = blockSize;
  _qualityLevel = qualityLevel;
  _minDifPercent = minDifPercent;
  _windowSize = cv::Size(windowSize, windowSize);
  _maxPyramideLevel = maxPyramideLevel;
}