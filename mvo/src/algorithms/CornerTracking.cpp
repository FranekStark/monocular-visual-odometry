#include "CornerTracking.hpp"

CornerTracking::CornerTracking()   {
}

void CornerTracking::detectFeatures(std::vector<cv::Point2f> &corner, const cv::Mat &image, int numberToDetect,
                                    const std::vector<cv::Point2f> &existingFeatures, cv::Rect2d &mask,
                                    bool forceDetection, double qualityLevel, double k, double blockSize, double minDiffPercent) {
  if (numberToDetect <= 0) {
    return;
  }
  //Create Mask
  cv::Mat maskImage(image.size(), CV_8U);
  maskImage = cv::Scalar(255); //First use all Pixels
  double imageDiag = sqrt(image.rows * image.rows + image.cols * image.cols);
  double mindistance = minDiffPercent * imageDiag;
  for (const auto & existingFeature : existingFeatures) {
    cv::circle(maskImage, existingFeature, (int)mindistance, cv::Scalar(0),
               -1); //Here no Feature Detection! //TODO: param
  }
  cv::rectangle(maskImage, mask, cv::Scalar(0), cv::FILLED);


  if (forceDetection) {
    qualityLevel = 0.001; //Set Quality Low, to Detect as much features as possible
  }
  cv::goodFeaturesToTrack(image, corner, numberToDetect, qualityLevel, mindistance, maskImage, blockSize,
                          bool(true),
                          k);  // Corners berechnen TODO: More params -> especially the detection Quality

  // Subpixel-genau:
  if (!corner.empty()) {
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria =
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(image, corner, winSize, zeroZone, criteria);
  }
}

void CornerTracking::trackFeatures(const std::vector<cv::Mat> &currentPyramide,
                                   const std::vector<cv::Mat> &previousPyramide,
                                   const std::vector<cv::Point2f> &prevFeatures,
                                   std::vector<cv::Point2f> &trackedFeatures,
                                   std::vector<unsigned char> &found,
                                   cv::Rect2d &mask,
                                   int maxPyramidLevel,
                                   cv::Size windowSize,
                                   double minDifPercent
                                   ) {
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
  cv::calcOpticalFlowPyrLK(previousPyramide, currentPyramide, prevFeatures, trackedFeatures, found, error, windowSize,
                           maxPyramidLevel, criteria, flags);  // TODO: more
  auto currentTopImage = std::vector<cv::Mat>(currentPyramide)[0];


  double imageDiag = sqrt(currentTopImage.rows * currentTopImage.rows + currentTopImage.cols * currentTopImage.cols);
  double mindistance = minDifPercent * imageDiag;

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

std::vector<cv::Mat> CornerTracking::createPyramide(cv::Mat image, cv::Size windowSize, int maxPyramideLevel) const {
  std::vector<cv::Mat> result;
  cv::buildOpticalFlowPyramid(image, result, windowSize, maxPyramideLevel);
  return result;
}

