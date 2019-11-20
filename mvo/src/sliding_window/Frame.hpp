#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <image_geometry/pinhole_camera_model.h>
#include "Feature.hpp"
#include <mutex>
#include <algorithm>
#include "../Utils.hpp"
#include <mvo/mvoConfig.h>

class Frame {
 private:
  friend class VisualisationUtils;
  std::vector<Feature> _features;
  const std::vector<cv::Mat> _imagePyramide;
  const image_geometry::PinholeCameraModel _cameraModel;

  cv::Vec3d _baseLine;
  double _scale;
  const cv::Matx33d _rotation;

  Frame *_preFrame;
  mutable std::mutex _lock;

  mvo::mvoConfig _parameters;
  ros::Time _timeStamp;

  void lock() const;
  void unlock() const;
 public:
  /**
   * Constructs a new Frame
   * @param imagePyramide the imagepyramide of this Frame
   * @param camerModel the ameramodel of the image
   * @param rotation the rotation of the camera
   * @param preFrame pointer to the frame before
   */
  Frame(std::vector<cv::Mat> imagePyramide, image_geometry::PinholeCameraModel camerModel, cv::Matx33d rotation, Frame * preFrame,  mvo::mvoConfig params, ros::Time timeStamp);

  virtual ~Frame() = default;
/**
 * Adds tracked Features from Frame before to the current Frame.
 * The Size of the three Paramsvector have to bee the same and on same position as the Size of the knowing Features in the Frame before
 *
 * @param trackedFeaturesNow A vector with the coordinates in camerapixelcoordinates of the Features
 * @param trackedFeaturesNowE A vector with the coordinate in eculieanprojected-Coordinates of the Features
 * @param found A vector wether the respectively Feature has been found
 */
  void addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                          const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                          const std::vector<unsigned char> &found);
/**
 * Adds this Features as 'new Features' to this Frame.
 * Logically they have no prefeature
 * @param features pixel-coordinate
 * @param featuresE image-coordinate
 */
  void addFeaturesToFrame(const std::vector<cv::Point2f> &features, const std::vector<cv::Vec3d> &featuresE);
/**
 * Retrieves corresponding Features in two ore more subsequently Frames.
 *
 * For this algorithm the PrefeatureCounter has to be set correct!
 *
 * @tparam T the Type of the Features
 * @param oldestFrame the oldest Frame
 * @param newestFrame the newest Frame
 * @param features the return of the features. Index 0, contains the newest. And the last Index the newest. The Featurevectors have to be empty!
 */
  template<typename T>
  static void getCorrespondingFeatures(const Frame &oldestFrame,
                                       const Frame &newestFrame,
                                       std::vector<std::vector<T> *> features) {

/*Calculate Depth and maximum Number Of Features  and check wether the params are okay*/
    //Lock Ingoing Frame:
    newestFrame.lock();
    unsigned int depth = 0;
    const Frame *frame = &newestFrame;
    auto outFeatures = features.begin();
    while (frame != &oldestFrame) {
      //Lock the Frames:
      depth++;
      assert(*outFeatures != nullptr);
      assert((*outFeatures)->size() == 0);
      (*outFeatures)->reserve(frame->_features.size());
      frame = frame->_preFrame;
      assert(frame != nullptr);
      frame->lock();
    }
    assert(depth > 0);
    assert(features.size() == (depth + 1));

/*Iterate through the Features and check wether are enough preFeatures and than get them*/

    for (auto newestFeature = newestFrame._features.begin(); newestFeature != newestFrame._features.end();
         newestFeature++) {
      if (newestFeature->_preFeatureCounter >= (depth)) {
        frame = &newestFrame;
        const Feature *feature = &(*newestFeature);
        unsigned int i = 0;
        do {
          outFeatures[i]->push_back(getFeatureLocation<T>(*feature));
          i++;
          frame = frame->_preFrame;
          feature = &frame->_features[feature->_preFeature];
        } while (i < depth);
        outFeatures[i]->push_back(getFeatureLocation<T>(*feature));
      }
    }

    //Unlock Frames;
    newestFrame.unlock();
    frame = &newestFrame;
    while (frame != &oldestFrame) {
      //Lock the Frames:
      frame = frame->_preFrame;
      frame->unlock();
    }
  }
/**
 * Returns the Image of specified Frame.
 * (Top of Image pyramid)
 *
 * @return the image
 */
  const cv::Mat &getImage();
/**
 * Retrieves Rotation of specific Frame
 *
 * @return the Rotation
 */
  const cv::Matx33d &getRotation();
/**
 * Retrieves the Cameramodell of specific Frame
 *
 *
 * @return the Cameramodell
 */
  const image_geometry::PinholeCameraModel &getCameraModel();
/**
 * Recieves the current tracked and detected Features in that Frame
 * @return the number of features
 */
  unsigned int getNumberOfKnownFeatures();
/**
 * Removes the Features from that Frame and tells that also the previous and following Frame
 * @param indices list of indeces of the to features to remove
 */
  void disbandFeatureConnection(const std::vector<unsigned int> &indices);
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
   * Retrieves corresponding Feature of two subsequent frames.
   * This Algorihtm also works, if the Prefeaturecounter isn't set correct yet.
   *
   * @tparam T type of the features
   * @param oldFrame the direct subdecessor of newFrame
   * @param newFrame the direct predecessor of oldFrame
   * @param oldFeatures corresponding old Feature locations
   * @param newFeatures corresponding new Featurelocatios
   */
  template<typename T>
  static void getCorrespondingFeatures(const Frame &oldFrame,
                                       const Frame &newFrame,
                                       std::vector<T> &oldFeatures,
                                       std::vector<T> &newFeatures) {
    newFrame.lock();
    oldFrame.lock();
    assert(newFrame._preFrame == &oldFrame);
    assert(oldFeatures.empty());
    assert(newFeatures.empty());

    oldFeatures.reserve(newFrame._features.size());
    newFeatures.reserve(newFrame._features.size());

    for (auto newFeature = newFrame._features.begin(); newFeature != newFrame._features.end(); newFeature++) {
      int preFeatureIndex = newFeature->_preFeature;
      if (preFeatureIndex >= 0) { //If valid PreFeature ID
        newFeatures.push_back(getFeatureLocation<T>(*newFeature));
        oldFeatures.push_back(getFeatureLocation<T>(oldFrame._features[preFeatureIndex]));
      }
    }
    oldFrame.unlock();
    newFrame.unlock();
  }
/**
   * Iterates through each feature and sets the correct Precounter based on ONLY the features in the Preframe.
   * So it is necesserry, that the Prefeatures have correct counters.
   *
   * It Simply it takes each Feature and if the Feature has a Prefeatureindex >= 0 (valid)
   * it takes the prefeaturecounter of the prefeature, increases it by one and safes it as the prefeaturecounter.
   *
   * @param frame relevant frame
   */
  void calculateFeaturePreCounter();
/**
 * Retrieves the baseline to previous of specific Frame
 * @return baseline to previous
 */
  cv::Vec3d getBaseLineToPrevious();

  /**
   * Retrieves the Scale of the baseline from this FRame to previous Frame
   * @return the scale
   */
  double getScaleToPrevious();

/**
   * Retrieves all known Features in specific Frame
   *
   * @tparam T theType of the Features
   *
   * @param features reference to an empty vector where the features will be placed
   */
  template<typename T>
  void getFeatures(std::vector<T> &features) {
    this->lock();
    //Check wether the Vector is empty
    assert(features.size() == 0);
    //Reserve needed memory
    features.reserve(this->_features.size());
    //Pushback all Features
    for (auto feature = this->_features.begin(); feature != this->_features.end(); feature++) {
      features.push_back(getFeatureLocation<T>(*feature));
    }
    this->unlock();
  }
/**
 * Retrieves the Imagepyramide of specific Frame
 *
 *
 * @return the ImagePyramid
 */
  const std::vector<cv::Mat> &getImagePyramid();
/**
 * Sets the  baseline to the previous Frame to specific Frame
 * @param baseLine the baseline
 */
  void setBaseLineToPrevious(const cv::Vec3d &baseLine);

  /**
   * Sets the scale of the baseLine from this to previous Frame
   * @param scale the scale
   */
  void setScaleToPrevious(double scale);

/**
   * Returns the Location of the Feature
   * @tparam T the Featurelocation Type
   * @return the Location
   */
  template<typename T>
  static const T &getFeatureLocation(const Feature &f);

  /**
   * Retrieves wether this Frame is the frist Frame (so it has no preframe).
   * @return wether its the first
   */
  bool isFirstFrame();

  /**
   * Retrieves the Parameters which the algorithm needs
   * @return reference to the parameter-set
   */
  const mvo::mvoConfig & getParameters();

  /**
   * Sets mew Parameters for this Frame
   * @param config the new Parameters
   */
  void setParameters(const mvo::mvoConfig & config);

  /**
   * Retrieves the timestamp of the Frame.
   * @return timestamp, of the the capturingtime of the image
   */
  ros::Time getTimeStamp();

};

#endif //FRAME_HPP