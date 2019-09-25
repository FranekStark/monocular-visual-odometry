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
  static void addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                                 const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                                 const std::vector<unsigned char> &found,
                                 Frame &frame);

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


  static void addFeaturesToFrame(Frame &frame, const std::vector<cv::Point2f> &features,
                                 const std::vector<cv::Vec3d> &featuresE);

/**
 * Retrieves the baseline to previous of specific Frame
 * @param frame the frame
 * @return baseline to previous
 */
  static const cv::Vec3d &getBaseLineToPrevious(const Frame &frame);

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
  static const cv::Mat getImagePyramid(Frame &frame);

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
 * The Frames have to be direct successors! (Source  is sucessor of target)
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
   * Iterates through all Features and fixes possible wrong featurecounter
   * which can occur when in pastFEatures the counters has been changed AFTER a new Frame has been created
   * @param frame regarding frame
   * @param the maximumDepth of the check
   */
  static void fixPreFeatureCounter(Frame &frame, unsigned int maximumDepth);

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

  void removeFeaturesFromFrame(std::vector<unsigned int> &featureIndeces, unsigned int past);

  bool isTemporaryFrame(unsigned int past) const;

  void exportMatlabData();

  /**
   * Returns a Pointer to the newest (last) Frame
   * @return pointer to the Frame
   */
  Frame *getNewestFrame();
};

#endif //SLIDING_WINDOW_HPP