#ifndef SLIDING_WINDOW_HPP
#define SLIDING_WINDOW_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>

#include "Frame.hpp"

class SlidingWindow {
 private:

 public:
  SlidingWindow();

  ~SlidingWindow();


/**
 * Adds tracked Features from Frame before to the current Frame.
 * The Size of the three Paramsvector have to bee the same and on same position as the Size of the knowing Features in the Frame before
 *
 * @param trackedFeaturesNow A vector with the coordinates in camerapixelcoordinates of the Features
 * @param trackedFeaturesNowE A vector with the coordinate in eculieanprojected-Coordinates of the Features
 * @param found A vector wether the respectively Feature has been found
 */
  static void addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                                 const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                                 const std::vector<unsigned char> &found,
                                 Frame &frame);

/**
 * Adds this Features as 'new Features' to specified Frame.
 * Logically they have no prefeature
 * @param frame the Frame
 * @param features pixel-coordinate
 * @param featuresE image-coordinate
 */
  static void addFeaturesToFrame(Frame &frame, const std::vector<cv::Point2f> &features,
                                 const std::vector<cv::Vec3d> &featuresE);

/**
 * Retrieves the baseline to previous of specific Frame
 * @param frame the frame
 * @return baseline to previous
 */
  static cv::Vec3d getBaseLineToPrevious(const Frame &frame);

/**
 * Retrieves corresponding Features in two ore more subsequently Frames.
 *
 * @tparam T the Type of the Features
 * @param oldestFrame the oldest Frame
 * @param newestFrame the newest Frame
 * @param features the return of the features. Index 0, contains the newest. And the last Index the newest. The Featurevectors have to be empty!
 */
  template<typename T>
  static void getCorrespondingFeatures(const Frame &oldestFrame,
                                       const Frame &newestFrame,
                                       std::vector<std::vector<T> *> features);
  /**
   * Retrieves all known Features in specific Frame
   *
   * @tparam T theType of the Features
   * @param frame the Frame to retrieve the Features
   * @param features reference to an empty vector where the features will be placed
   */
  template<typename T>
  static void getFeatures(const Frame &frame, std::vector<T> &features);

  /**
   * Returns the Location of the Feature
   * @tparam T the Featurelocation Type
   * @return the Location
   */
  template<typename T>
  static const T &getFeatureLocation(const Feature &f);

  /**
   * Retrieves Rotation of specific Frame
   * @param frame the Frame
   * @return the Rotation
   */
  static const cv::Matx33d &getRotation(const Frame &frame);

  /**
   * Retrieves the Cameramodell of specific Frame
   *
   * @param frame the Frame
   * @return the Cameramodell
   */
  static const image_geometry::PinholeCameraModel &getCameraModel(const Frame &frame);

  /**
   * Retrieves the Imagepyramide of specific Frame
   *
   * @param frame the Frame
   * @return the ImagePyramid
   */
  static const cv::Mat & getImagePyramid(Frame &frame);

  /**
   * Recieves the current tracked and detected Features in that Frame
   * @param frame the frame
   * @return the number of features
   */
  static unsigned int getNumberOfKnownFeatures(Frame &frame);

  /**
   * Removes the Features from that Frame and tells that also the previous and following Frame
   * @param indices list of indeces of the to features to remove
   */
  static void disbandFeatureConnection(const std::vector<unsigned int> &indices, Frame &frame);

  /**
   * Sets the 'temporarelly needed' baseline to the previous Frame to specific Frame
   * @param frame specific Frame
   * @param baseLine the baseline
   */
  static void setBaseLineToPrevious(Frame &frame, const cv::Vec3d &baseLine);

/**
 * Updates the featureLocation and add the new Features.
 * The Rotation and image also will be updated.
 *
 * The Frames have to be direct successors! (Source  is successor of target)
 *
 * Technically it Replaces target with source AND updates the References
 *
 * @param targetFrame the frame which becomes updated
 * @param sourceFrame the frame where the data comes from
 */
  static void updateFrame(Frame &targetFrame, Frame &sourceFrame);

  /**
   * Merges the feature Locations. Which means that new Features will be added but the locations of existing features won't be updated.
   * Also image and rotation won't be updated.
   * In source Frame there will be connections to the features added.
   * If there are more Frames between source and target than the Frames in the middle will be deleted.
   *
   * The source has to be timely AFTER the target
   *
   * @param targetFrame the frame into which is merged
   * @param sourceFrame the frame from where is merge
   */
  static void mergeFrame(Frame &targetFrame, Frame &sourceFrame);

  /**
   * Iterates through each feature and sets the correct Precounter based on ONLY the features in the Preframe.
   * So it is necesserry, that the Prefeatures have correct counters.
   *
   * It Simply it takes each Feature and if the Feature has a Prefeatureindex >= 0 (valid)
   * it takes the prefeaturecounter of the prefeature, increases it by one and safes it as the prefeaturecounter.
   *
   * @param frame relevant frame
   */
  static void calculateFeaturePreCounter(Frame &frame);

  /**
   * Returns the Image of specified Frame.
   * (Top of Image pyramid)
   *
   * @param frame the frame
   * @return the image
   */
  static const cv::Mat & getImage(Frame &frame);

};

#endif //SLIDING_WINDOW_HPP