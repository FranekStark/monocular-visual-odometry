//
// Created by franek on 24.09.19.
//

#include "Merger.h"

Merger::Merger(PipelineStage &precursor,
               unsigned int outGoingChannelSize,
               double sameThreshold,
               double movementThreshold) :
    PipelineStage( precursor, outGoingChannelSize),
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
  }

  //Calculate Disparity between preFrame and the new Frame
  std::vector<cv::Vec3d> preCorrespondingFeatures, newCorrespondingFeatures, newCorrespondingFeaturesUnrotated;
  SlidingWindow::getCorrespondingFeatures<cv::Vec3d>(*_preFrame,
                                                     *newFrame,
                                                     {&preCorrespondingFeatures, &newCorrespondingFeatures});
  auto preRotation = SlidingWindow::getRotation(*_preFrame);
  auto nowRotation = SlidingWindow::getRotation(*newFrame);
  auto diffRotation = preRotation.t() * nowRotation;
  FeatureOperations::unrotateFeatures(newCorrespondingFeatures, newCorrespondingFeaturesUnrotated, diffRotation);
  auto disparity = FeatureOperations::calcDisparity(preCorrespondingFeatures, newCorrespondingFeaturesUnrotated);
#ifdef DEBUGIMAGES
  if (newFrame->_preFrame != nullptr) {
    cv::Mat image = SlidingWindow::getImage(*newFrame).clone();
    VisualisationUtils::drawFeaturesUnrotated(*newFrame, image);
  }
#endif
  //Casedifferntation based on amount of the difference
  if (disparity <= _sameDisparityThreshold) { //Threat the new Frame as if where on the SAME position as preFrame
    //Merge the new Frame onto the preFrame
    SlidingWindow::mergeFrame(*_preFrame, *newFrame);
    //That will automaticly delete the Frame between, so:
    ///keep_preFrame = _preFrame;
    _keepFrame = newFrame;
    return nullptr; //Hold PipeLine
  } else if (disparity <= _movementDisparityThreshold) { //Not enough disparity, HOLD Pipeline
    //Update the keep Frame with the new Frame, if there is one
    if (_keepFrame != nullptr) {
      SlidingWindow::updateFrame(*_keepFrame, *newFrame);
    }
    //Above deletes the _keepFrame or maybe there was no _keepFrame so:
    _keepFrame = newFrame;
    return nullptr; //Hold PipeLine
  } else { //Enough Disparity
    //Update the keept, if there is one:
    if (_keepFrame != nullptr) {
      SlidingWindow::updateFrame(*_keepFrame, *newFrame);
    }
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

