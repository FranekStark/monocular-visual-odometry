#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

class CornerTracker {
 private:
  int _blockSize;
  int _apertureSize;
  double _k;
  int _thresh;
  double _minDifPercent;
  double _qualityLevel;
  cv::Size _windowSize;
  int _maxPyramideLevel;

  cv::Mat _beforePyramide;

 public:
  CornerTracker();

  ~CornerTracker();

  void detectFeatures(std::vector<cv::Point2f> &corner, const cv::Mat &image, int numberToDetect,
                      const std::vector<cv::Point2f> &existingFeatures, cv::Rect2d &mask, bool forceDetection);

  void trackFeatures(const cv::Mat &currentPyramide,
                     const std::vector<cv::Point2f> &prevFeatures, std::vector<cv::Point2f> &trackedFeatures,
                     std::vector<unsigned char> &found, cv::Rect2d &mask);

  void setCornerDetectorParams(int blockSize, double minDifPercent, double qualityLevel, int windowSize,
                               int maxPyramideLevel);
  /**
   * Creates a pyramide representation of the image with current params of CornerTracker
   * @param image the image from which the pyramide will be constrcuted (maybe reused in pyramide)
   * @return the pyramide
   */
  cv::Mat createPyramide(cv::Mat image) const;

};
