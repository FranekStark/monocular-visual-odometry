//
// Created by franek on 24.09.19.
//

#include "TrackerDetector.hpp"

TrackerDetector::TrackerDetector(PipelineStage &precursor,
                                 unsigned int outGoingChannelSize,
                                 CornerTracking &cornerTracking) :
    PipelineStage(&precursor, outGoingChannelSize),
    _prevFrame(nullptr),
    _cornerTracking(cornerTracking)
   {
#ifdef DEBUGIMAGES
  cv::namedWindow("TrackerImage", cv::WINDOW_NORMAL);
  cv::moveWindow("TrackerImage", 3360,895);
  cv::resizeWindow("TrackerImage", 960,988);
  cv::startWindowThread();
#endif

}

Frame *TrackerDetector::stage(Frame *newFrame) {
  //In Case the previous Frame is null only detect new Features
  if (_prevFrame == nullptr) {
    detect(*newFrame, newFrame->getParameters().numberOfFeatures);
  } else { //Else track and then detect
    track(*newFrame);
    int numberToDetect = newFrame->getParameters().numberOfFeatures - newFrame->getNumberOfKnownFeatures();
    detect(*newFrame, numberToDetect);
  }
#ifdef DEBUGIMAGES
  cv::Mat image;
  cv::cvtColor(newFrame->getImage(), image, cv::COLOR_GRAY2BGR);
  VisualisationUtils::drawFeatures(*newFrame, image);
  VisualisationUtils::drawRect(image, getShipMask(newFrame->getImage().size()));
  cv::imshow("TrackerImage", image);
  cv::waitKey(10);
#endif
  //Pass through the new Frame
  _prevFrame = newFrame;
  return newFrame;
}

void TrackerDetector::track(Frame &newFrame) {
  std::vector<cv::Point2f> prevFeatures, trackedFeatures;
  /*Get previous Features*/
  _prevFrame->getFeatures(prevFeatures);
  /*Calculate Difference rotation*/
  auto prevRotation = _prevFrame->getRotation();
  auto nowRotation = newFrame.getRotation();
  auto diffRotation = prevRotation.t() * nowRotation;
  /*Estimate new location regarding to rotation*/
  std::vector<cv::Vec3d> prevFeaturesE, prevFeaturesERotated;
  //Project the Features to Rotate them
  FeatureOperations::euclidNormFeatures(prevFeatures, prevFeaturesE, _prevFrame->getCameraModel());
  //Rotate them onto new Position
  FeatureOperations::unrotateFeatures(prevFeaturesE, prevFeaturesERotated, diffRotation.t());
  //Reproject the rotated Features
  FeatureOperations::euclidUnNormFeatures(prevFeaturesERotated, trackedFeatures, newFrame.getCameraModel());
  /*Track the Features (with guess from Rotation)*/
  std::vector<unsigned char> found;
  auto camInfo = newFrame.getCameraModel();
  auto shipMask = getShipMask(newFrame.getImage().size());
  /*Check which Frames parameter we use*/
  int pyramidLevel = std::min(newFrame.getParameters().pyramidDepth, _prevFrame->getParameters().pyramidDepth);//Depends on the flatter pyramide
  int winSizeX = std::min(newFrame.getParameters().windowSizeX, _prevFrame->getParameters().windowSizeX);  //Also Depends on the smaller
  int winSizeY = std::min(newFrame.getParameters().windowSizeY, _prevFrame->getParameters().windowSizeY);   //   "

  _cornerTracking.trackFeatures(newFrame.getImagePyramid(),
                               _prevFrame->getImagePyramid(),
                                prevFeatures,
                                trackedFeatures,
                                found,
                                shipMask, pyramidLevel, cv::Size(winSizeX, winSizeY), newFrame.getParameters().mindDiffPercent);
  /*Project them to Euclidean*/
  std::vector<cv::Vec3d> trackedFeaturesE;
  FeatureOperations::euclidNormFeatures(trackedFeatures, trackedFeaturesE, newFrame.getCameraModel());
  /*Add them to the Frame*/
  newFrame.addTrackedFeatures(trackedFeatures, trackedFeaturesE, found);
}

void TrackerDetector::detect(Frame &newFrame, unsigned int number) {
  /*Detect new Features*/
  std::vector<cv::Point2f> existingFeatures, newFeatures;
  newFrame.getFeatures(existingFeatures); //get Current Existing Features
  auto camInfo = newFrame.getCameraModel();
  auto shipMask = getShipMask(newFrame.getImage().size());
  _cornerTracking.detectFeatures(newFeatures,
                                 std::vector<cv::Mat>(newFrame.getImagePyramid())[0],
                                 number,
                                 existingFeatures,
                                 shipMask,
                                 false, newFrame.getParameters().qualityLevel, newFrame.getParameters().k, newFrame.getParameters().blockSize, newFrame.getParameters().mindDiffPercent);
  /*Convert them and Put them back*/
  std::vector<cv::Vec3d> newFeaturesE;
  FeatureOperations::euclidNormFeatures(newFeatures, newFeaturesE, newFrame.getCameraModel());
  newFrame.addFeaturesToFrame(newFeatures, newFeaturesE);
}
TrackerDetector::~TrackerDetector() {
  cv::destroyWindow("TrackerImage");
}
cv::Rect2d TrackerDetector::getShipMask(const cv::Size& imageSize) {

  /* Mask, where Ship is In Image*/
  cv::Rect2d shipMask((imageSize.width / 2) - (1.6 / 16.0) * imageSize.width,
                      imageSize.height - (6.5 / 16.0) * imageSize.height, (3.2 / 16.0) * imageSize.width,
                      (6.5 / 16.0) * imageSize.width);
  return shipMask;

}


