//
// Created by franek on 24.09.19.
//

#include "Merger.h"

Merger::Merger(PipelineStage &precursor,
               unsigned int outGoingChannelSize) :
    PipelineStage(&precursor, outGoingChannelSize),
    _preFrame(nullptr),
    _keepFrame(nullptr) {
#ifdef DEBUGIMAGES
  cv::namedWindow("MergerImage", cv::WINDOW_NORMAL);
  cv::moveWindow("MergerImage", 2400, 895);
  cv::resizeWindow("MergerImage", 960, 988);
  cv::startWindowThread();
#endif
}

Frame *Merger::stage(Frame *newFrame) {
  if (_preFrame == nullptr) { //In Case its the first Frame -> FastPipe
    _preFrame = newFrame;
    return _preFrame;
  } else if (_keepFrame == nullptr) {
    _keepFrame = newFrame;
  } else {
    //In every 'disparity-Case' we do the Update to the _keeptFrame, cause we need new Locations
    Frame::updateFrame(*_keepFrame, *newFrame);
    _keepFrame = newFrame; //_keepFrame is deleted through Update-Function
  }

  //Calculate Disparity between keepFrame and preFrame
  std::vector<cv::Vec3d> preCorrespondingFeatures, newCorrespondingFeatures,
      newCorrespondingFeaturesUnrotated;

  Frame::getCorrespondingFeatures(*_preFrame, *_keepFrame, preCorrespondingFeatures, newCorrespondingFeatures);


  //Rotate Features to previousframe to only measure the difference caused by movement not rotation
  auto preRotation = _preFrame->getRotation();
  auto nowRotation = newFrame->getRotation();
  auto diffRotation = preRotation.t() * nowRotation;
  FeatureOperations::unrotateFeatures(newCorrespondingFeatures, newCorrespondingFeaturesUnrotated, diffRotation);
  auto disparity = FeatureOperations::calcDisparity(preCorrespondingFeatures, newCorrespondingFeaturesUnrotated);
#ifdef DEBUGIMAGES
  if (!(newFrame->isFirstFrame())) {
    std::vector<cv::Point2f> preCorespF, nowCorespF, nowCorespFU;
    FeatureOperations::euclidUnNormFeatures(preCorrespondingFeatures,
                                            preCorespF,
                                            _preFrame->getCameraModel());
    FeatureOperations::euclidUnNormFeatures(newCorrespondingFeatures,
                                            nowCorespF,
                                            newFrame->getCameraModel());
    FeatureOperations::euclidUnNormFeatures(newCorrespondingFeaturesUnrotated,
                                            nowCorespFU,
                                            newFrame->getCameraModel());
    cv::Mat image;
    cv::cvtColor(newFrame->getImage(), image, cv::COLOR_GRAY2BGR);
    if (disparity > newFrame->getParameters().movementDisparityThreshold) {
      image = cv::Scalar(255, 255, 255);
    }
    VisualisationUtils::drawFeaturesUnrotated(image, preCorespF, nowCorespF, nowCorespFU);
    cv::putText(image,
                "Disparity: " + std::to_string(disparity),
                cv::Point(10, 40),
                cv::FONT_HERSHEY_PLAIN,
                4,
                cv::Scalar(0, 255, 255));
    cv::imshow("MergerImage", image);
    cv::waitKey(10);
  }
#endif
  //Casedifferntation based on amount of the difference
  if (disparity <= newFrame->getParameters().sameDisparityThreshold) { //Threat the new Frame as if where on the SAME position as preFrame
    //Merge the new Frame onto the preFrame
    Frame::mergeFrame(*_preFrame, *newFrame);
    LOG_DEBUG("Merged " << newFrame << " into " << _preFrame);
    return nullptr; //Hold PipeLine
  } else if (disparity <= newFrame->getParameters().movementDisparityThreshold) { //Not enough disparity, HOLD Pipeline
    return nullptr; //Hold PipeLine
  } else { //Enough Disparity
    //Calculate the correct prefeaturecounter:
    newFrame->calculateFeaturePreCounter();
    //Pipethrough
    _preFrame = newFrame;
    _keepFrame = nullptr; //Wait here for a maybe new Frame
    return newFrame;
  }

}

Merger::~Merger() {
#ifdef DEBUGIMAGES
  cv::destroyWindow("MergerImage");
#endif
}

