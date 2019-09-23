#ifndef SLIDING_WINDOW_HPP
#define SLIDING_WINDOW_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>

#include "Frame.hpp"

class SlidingWindow {
 private:
  /*Maximum Number of Features*/
  unsigned int _maxFeatures;
  /*Size of the SlidingWindow (how many Windows will be kept) */
  unsigned int _length;
  /*Number Of current Frames*/
  unsigned int _frameCounter;
  /*Last (timely) Window in the Past, which is still there*/
  Frame *_frameNow;

  /**
   * 0 means the Window NOW/Current
   * If the Window is not available, then nullptr
   */
  Frame &getFrame(unsigned int past) const;

 public:
  SlidingWindow(int len, unsigned int features);

  ~SlidingWindow();


  /**
 * Creates a new Frame or Updates Current if the current Frame is only "TEMP" and not "PERSIST"
 * @param image the image of the current Frame
 * @param cameraModel the cameraModel reagrding the current image
 */
  void newFrame(cv::Mat image, image_geometry::PinholeCameraModel cameraModel);

  void newFrame(const std::vector<cv::Point2f> &trackedFeaturesNow,
                const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                const std::vector<unsigned char> &found, cv::Mat image);

/**
 * Adds tracked Features from Frame before to the current Frame.
 * The Size of the three Paramsvector have to bee the same, as the Size of the knowing Features in the Frame before
 *
 * @param trackedFeaturesNow A vector with the coordinates in camerapixelcoordinates of the Features
 * @param trackedFeaturesNowE A vector with the coordinate in eculieanprojected-Coordinates of the Features
 * @param found A vector wether the respectively Feature has been found
 */
  void addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                          const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                          const std::vector<unsigned char> &found);



  /**
   * This updates the know Features by the current Frame.
   * All the Parameter Vectors have to have the same Size and the same size as the current known features.
   *
 * @param trackedFeaturesNow A vector with the coordinates in camerapixelcoordinates of the Features
 * @param trackedFeaturesNowE A vector with the coordinate in eculieanprojected-Coordinates of the Features
 * @param found A vector wether the respectively Feature is still there
   */
  void updateFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                      const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                      const std::vector<unsigned char> &found);


  void addNewFeaturesToFrame(const std::vector<cv::Point2f> &features,
                             const std::vector<cv::Vec3d> &featuresE, unsigned int past);

  void addNewFeaturesToBeforeFrame(const std::vector<cv::Point2f> &features,
                                   const std::vector<cv::Vec3d> &featuresE);

  void persistCurrentFrame();

  /**
   * past = 0, means Current Windows Features
   * When no Window at this time, nullptr is returned
   */
  void getFeatures(unsigned int past, std::vector<cv::Point2f> &features) const;

  void getFeatures(unsigned int past, std::vector<cv::Vec3d> &features) const;

  const cv::Mat getImage(unsigned int past) const;

  cv::Vec3d &getPosition(unsigned int past) const;

  cv::Matx33d &getRotation(unsigned int past) const;

  void setPosition(const cv::Vec3d &position, unsigned int past);

  void setRotation(const cv::Matx33d &rotation, unsigned int past);

  image_geometry::PinholeCameraModel &getCameraModel(unsigned int past);

/**
 * Gives Back Corresponding Featurelocations between two Windows.
 * Window2 is timely after Window1. While 0 means NOW, and the Maximum is _length.
 * Usually Window 2 is 0.
 */
  void
  getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Point2f> &features1,
                           std::vector<cv::Point2f> &features2) const;

  void
  getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Vec3d> &features1,
                           std::vector<cv::Vec3d> &features2) const;

  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                std::vector<std::vector<cv::Vec3d> *> features) const;

  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                std::vector<std::vector<cv::Point2f> *> features) const;

  unsigned int getNumberOfCurrentTrackedFeatures() const;

  void removeFeatureFromCurrentWindow(const cv::Vec3d &feature);

  void removeFeaturesFromWindow(const std::vector<unsigned igend int> featureIndeces, unsigned int past);

  bool isTemporaryFrame(unsigned int past) const;

  void exportMatlabData();
};

#endif //SLIDING_WINDOW_HPP