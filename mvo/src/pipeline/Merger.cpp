//
// Created by franek on 24.09.19.
//

#include "Merger.h"

Merger::Merger(PipelineStage &precursor,
               unsigned int outGoingChannelSize,
               double sameThreshold,
               double movementThreshold) :
    PipelineStage(&precursor, outGoingChannelSize),
    _preFrame(nullptr),
    _keepFrame(nullptr),
    _sameDisparityThreshold(sameThreshold),
    _movementDisparityThreshold(movementThreshold) {
#ifdef DEBUGIMAGES
  cv::namedWindow("MergerImage", cv::WINDOW_NORMAL);
#endif
}

Frame *Merger::stage(Frame *newFrame) {
if (_preFrame == nullptr) { //In Case its the first Frame -> FastPipe
    _preFrame = newFrame;
    return _preFrame;
  }else if(_keepFrame == nullptr){
    _keepFrame = newFrame;
  }else{
    //In every 'disparity-Case' we do the Update to the _keeptFrame, cause we need new Locations
    SlidingWindow::updateFrame(*_keepFrame, *newFrame);
    _keepFrame = newFrame; //_keepFrame is deleted through Update-Function
  }

  //Calculate Disparity between keepFrame and preFrame
  std::vector<cv::Vec3d> preCorrespondingFeatures, newCorrespondingFeatures,
      newCorrespondingFeaturesUnrotated;

  SlidingWindow::getCorrespondingFeatures(*_preFrame, *_keepFrame, preCorrespondingFeatures, newCorrespondingFeatures);

  //Rotate Features to previousframe to only measure the difference caused by movement not rotation
  auto preRotation = SlidingWindow::getRotation(*_preFrame);
  auto nowRotation = SlidingWindow::getRotation(*newFrame);
  auto diffRotation = preRotation.t() * nowRotation;
  FeatureOperations::unrotateFeatures(newCorrespondingFeatures, newCorrespondingFeaturesUnrotated, diffRotation);
  auto disparity = FeatureOperations::calcDisparity(preCorrespondingFeatures, newCorrespondingFeaturesUnrotated);
#ifdef DEBUGIMAGES
  if (newFrame->_preFrame != nullptr) {
    std::vector<cv::Point2f> preCorespF, nowCorespF, nowCorespFU;
    FeatureOperations::euclidUnNormFeatures(preCorrespondingFeatures,
                                            preCorespF,
                                            SlidingWindow::getCameraModel(*_preFrame));
    FeatureOperations::euclidUnNormFeatures(newCorrespondingFeatures,
                                            nowCorespF,
                                            SlidingWindow::getCameraModel(*newFrame));
    FeatureOperations::euclidUnNormFeatures(newCorrespondingFeaturesUnrotated,
                                            nowCorespFU,
                                            SlidingWindow::getCameraModel(*newFrame));

    cv::Mat image;
    cv::cvtColor(SlidingWindow::getImage(*newFrame), image, cv::COLOR_GRAY2BGR);
    VisualisationUtils::drawFeaturesUnrotated(image, preCorespF, nowCorespF, nowCorespFU);
    cv::imshow("MergerImage", image);
    cv::waitKey(10);
  }
#endif
  //Casedifferntation based on amount of the difference
  if (disparity <= _sameDisparityThreshold) { //Threat the new Frame as if where on the SAME position as preFrame
    //Merge the new Frame onto the preFrame
    SlidingWindow::mergeFrame(*_preFrame, *newFrame);
    return nullptr; //Hold PipeLine
  } else if (disparity <= _movementDisparityThreshold) { //Not enough disparity, HOLD Pipeline
    return nullptr; //Hold PipeLine
  } else { //Enough Disparity
    //Calculate the correct prefeaturecounter:
    SlidingWindow::calculateFeaturePreCounter(*newFrame);
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

