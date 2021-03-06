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
  cv::namedWindow("mvocv-MergerImage", cv::WINDOW_NORMAL);
  cv::moveWindow("mvocv-MergerImage", -36, 44);
  cv::resizeWindow("mvocv-MergerImage", 638, 475);
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
  auto timeDiff =  _keepFrame->getTimeStamp() - _preFrame->getTimeStamp();
  auto neededTimeDiff = ros::Duration(1.0/newFrame->getParameters().mergeFrequency);
  bool enoughTime = timeDiff >= neededTimeDiff;

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
    cv::Mat imagePre;
    cv::cvtColor(_preFrame->getImage(), imagePre, cv::COLOR_GRAY2BGR);
    cv::addWeighted(image, 0.5, imagePre, 0.5, 0.0, image);

    VisualisationUtils::drawFeaturesUnrotated(image, preCorespF, nowCorespF, nowCorespFU);
    auto brightnessVector = cv::mean(image.colRange(10, 100).rowRange(40, 100));
    double brightness = (brightnessVector[0] + brightnessVector[1] + brightnessVector[2]) / 3; //mean the three channels
    auto color = cv::Scalar(255, 255, 255);
    std::string hint = "";
    if (disparity > newFrame->getParameters().movementDisparityThreshold) {
      hint += " locOK";
    }
    if(enoughTime){
      hint += " tOK";
    }
    if (brightness > 125) {
      color = cv::Scalar(0, 0, 0);
    }
    cv::putText(image,
                "Disparity: " + std::to_string(disparity) + hint,
                cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                2,
                color,
                5);

    if((enoughTime || !newFrame->getParameters().useMergeFrequency) && disparity > newFrame->getParameters().movementDisparityThreshold) {
      image = cv::Scalar(255, 255, 255);
    }
    //Show Angle-Difference:
    double angle = acos((cv::trace(diffRotation) -1)/ 2) * 180.0 / M_PI;
    cv::putText(image, "Angle: " + std::to_string(angle) + " deg", cv::Point( 40, image.rows - 100), cv::FONT_HERSHEY_SIMPLEX,
                2,
                color,
                5);
    cv::imshow("mvocv-MergerImage", image);
    cv::waitKey(10);
  }
#endif


  //Casedifferntation based on amount of the difference
  if (disparity
      <= newFrame->getParameters().sameDisparityThreshold) { //Threat the new Frame as if where on the SAME position as preFrame
    //Merge the new Frame onto the preFrame
    Frame::mergeFrame(*_preFrame, *newFrame);
    LOG_DEBUG("Merged " << newFrame << " into " << _preFrame);
    return nullptr; //Hold PipeLine
  } else if (disparity <= newFrame->getParameters().movementDisparityThreshold
      || (!enoughTime
          && newFrame->getParameters().useMergeFrequency)) { //Not enough feature disparity OR not enough time disparity, HOLD Pipeline
    return nullptr;//Hold PipeLine
  } else { //Enough Disparity
    //Calculate the correct prefeaturecounter:
    newFrame->calculateFeaturePreCounter();
#ifdef RATINGDATA
    newFrame->_infos.MERGER_disparity = disparity;
#endif
    //Pipethrough
    _preFrame = newFrame;
    _keepFrame = nullptr; //Wait here for a maybe new Frame
    return newFrame;
  }

}

Merger::~Merger() {
#ifdef DEBUGIMAGES
  cv::destroyWindow("mvocv-MergerImage");
#endif
}

