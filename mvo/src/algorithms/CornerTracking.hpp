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

  /**
   * Detects new Features(Corners) in given image with respect to already known features (doesnt tracks them  again)
   *
   * Features are Harrison-Corners!
   *
   * @param corner the (frist empty!) vector where the features will be put into
   * @param image the image on which to detect the features
   * @param numberToDetect the number of features to detect
   * @param existingFeatures existing features, which wonÄt be detected again
   * @param mask a mask where also no features will be detected
   * @param forceDetection when this flag is set to true the 'qualitylevel' is ignored to detect all features specfied in 'numberToDetect'
   * @param qualityLevel the minimum quality of the detected features (depends on the quality of the first feature)
   * @param k free harrison corner parameter
   * @param blockSize the block siz ewhere harrison corner will integrate
   * @param minDiffPercent percent of the image size which features will have as distance (also to features in 'existing features')
   */
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

  /**
   * Trackes known features on one image to another image.
   * The images could be image-pyramids.
   *
   * This is a Lucas-Kanade-Pyramide-Tracker!
   *
   * @param currentPyramide the image(pyramide) where the features have to be tracked to
   * @param previousPyramide the image(pyramide) where the features are known
   * @param prevFeatures the known feature positions
   * @param trackedFeatures a forecast where the features could be or an empty vector where the tracked features will put in
   * @param found for each tracked feature there is an entry in here where '1'=found and '0'=not found
   * @param mask a mask where no feature is tracked to
   * @param maxPyramidLevel the size of the pyramidedepth (take care that the inputimages need to have a depth of this or higher)
   * @param windowSize the window Size where to search (on each pyramide level)
   * @param minDifPercent percent of the image size which features will have as distance (the algorithm will merge to close features together)
   */
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
