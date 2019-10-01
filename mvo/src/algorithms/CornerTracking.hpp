#ifndef CORNER_TRACKING_HPP
#define CORNER_TRACKING_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

class CornerTracking {
 private:

 public:
  CornerTracking();

  ~CornerTracking() = default;

  void detectFeatures(std::vector<cv::Point2f> &corner,
                      const cv::Mat &image,
                      int numberToDetect,
                      const std::vector<cv::Point2f> &existingFeatures,
                      cv::Rect2d &mask,
                      bool forceDetection,
                      double qualityLevel,
                      double k,
                      double blockSize,
                      double minDiffPercent);

  void trackFeatures(const std::vector<cv::Mat> &currentPyramide,
                     const std::vector<cv::Mat> &previousPyramide,
                     const std::vector<cv::Point2f> &prevFeatures,
                     std::vector<cv::Point2f> &trackedFeatures,
                     std::vector<unsigned char> &found,
                     cv::Rect2d &mask,
                     int maxPyramidLevel,
                     cv::Size windowSize,
                     double minDifPercent);

  /**
   * Creates a pyramide representation of the image with current params of CornerTracker
   * @param image the image from which the pyramide will be constrcuted (maybe reused in pyramide)
   * @return the pyramide
   */
  std::vector<cv::Mat> createPyramide(cv::Mat image, cv::Size windowSize, int maxPyramideLevel) const;

};

#endif //CORNER_TRACKING_HPP
