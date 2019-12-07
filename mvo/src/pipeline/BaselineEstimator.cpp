//
// Created by franek on 25.09.19.
//

#include "BaselineEstimator.hpp"

BaselineEstimator::BaselineEstimator(PipelineStage &precursor,
                                     unsigned int out_going_channel_size,
                                     EpipolarGeometry &epipolarGeometry) : PipelineStage(&precursor,
                                                                                         out_going_channel_size),
                                                                           _prevFrame(nullptr),
                                                                           _epipolarGeometry(epipolarGeometry),
                                                                           _baseLine(1) {
#ifdef DEBUGIMAGES
  cv::namedWindow("mvocv-EstimatorImage", cv::WINDOW_NORMAL);
  cv::moveWindow("mvocv-EstimatorImage", 24, 551);
  cv::resizeWindow("mvocv-EstimatorImage", 1139, 524);
  cv::startWindowThread();
#endif
}
Frame *BaselineEstimator::stage(Frame *newFrame) {
  if (_prevFrame == nullptr) { //If it is the first Frame
    newFrame->setBaseLineToPrevious(cv::Vec3d(0, 0, 0));
    newFrame->setScaleToPrevious(0);
#ifdef RATINGDATA
    newFrame->_infos.ESTIMATED_baseline = newFrame->getBaseLineToPrevious();
    newFrame->_infos.RANSAC_outsortet_features = 0;
    newFrame->_infos.RANSAC_probability = 0;
#endif
  } else {
    /* Correct Connections, cause by "disband" they wil be wrong*/
    newFrame->calculateFeaturePreCounter();
    /* Get CorrespondingFeatures */
    std::vector<cv::Vec3d> beforeCorespFeaturesE, thisCorespFeaturesUnrotatedE, thisCorespFeaturesE;
    Frame::getCorrespondingFeatures<cv::Vec3d>(*_prevFrame,
                                               *newFrame,
                                               beforeCorespFeaturesE, thisCorespFeaturesE);
    /* Unroate the Features */
    auto beforeRotaton = _prevFrame->getRotation();
    auto thisRotation = newFrame->getRotation();
    auto diffRotation = beforeRotaton.t() * thisRotation;
    FeatureOperations::unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesUnrotatedE, diffRotation);
    /* First Guess of the Direction-BaseLine */
    std::vector<unsigned int> inlier, outlier;

    //THRESHOLD(cos(3.0 * PI / 180.0)),
    double threshold = std::cos(newFrame->getParameters().thresholdOutlier * M_PI / 180.0);
    auto baseLine = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE,
                                                       thisCorespFeaturesUnrotatedE,
                                                       inlier,
                                                       newFrame->getParameters().bestFitProbability,
                                                       threshold);
    /*Special Algorithm to get Outlier Indeces from Inlier*/
    std::sort(inlier.begin(), inlier.end());
    auto inlierIT = inlier.begin();
    for (unsigned int i = 0; i < thisCorespFeaturesE.size(); i++) {
      if (inlierIT
          == inlier.end()) { //In that Case every following Index is an outlier, cause the inlier list is "empty"

        continue;
      }
      if (i < *inlierIT) { //Then the Value is an outlier, cause we start at lowest inlier value
        outlier.push_back(i);
      } else if (i == *inlierIT) {
        inlierIT++; //Rise both
      } else {// i > inlierIT
        ROS_ERROR("That case shouldn't match!\n\r");
      }
    }
    /*Remove Outlier*/
    newFrame->disbandFeatureConnection(outlier); //TODO: das Funktioniert nur, weil er wissen über den Tracker hat
    /* Transform the relative BaseLine into WorldCoordinates */
    baseLine = beforeRotaton * baseLine;

    //Detect a wring BAseline
    //Vote for the sign of the Baseline, which generates the feweset negative Gradients
    std::vector<double> depths, depthsNegate;
    auto bnegate = -1.0 * baseLine;
    std::vector<cv::Vec3d> prevFeatureD, nowFeatureD;
    Frame::getCorrespondingFeatures(*_prevFrame, *newFrame, prevFeatureD, nowFeatureD);
    FeatureOperations::calcProjectionsAngleDiff(depths,
                                                prevFeatureD,
                                                nowFeatureD,
                                                _prevFrame->getRotation(),
                                                newFrame->getRotation(),
                                                baseLine);
    FeatureOperations::calcProjectionsAngleDiff(depthsNegate,
                                                prevFeatureD,
                                                nowFeatureD,
                                                _prevFrame->getRotation(),
                                                newFrame->getRotation(),
                                                bnegate);
    double negCountb = 0;
    double negCountbnegate = 0;
    assert(depths.size() == depthsNegate.size());
    auto depthIt = depths.begin();
    auto depthnegateIt = depthsNegate.begin();
    while (depthIt != depths.end()) {

      if (((abs(180 - (*depthIt * 180.0 / M_PI) )) > (newFrame->getParameters().negativeDegreesThreshold)
          && abs(180 - (*depthnegateIt  * 180.0 / M_PI)) > (newFrame->getParameters().negativeDegreesThreshold))) {
        negCountb += cos(0.5 * *depthIt);
        negCountbnegate += cos(0.5 * *depthnegateIt);
      }

/*
      ROS_INFO_STREAM(
          "depth: " << *depthIt * 180.0 / M_PI << " (" << cos(0.5 * *depthIt) << ") | " << *depthnegateIt * 180.0 / M_PI
                    << " (" << cos(0.5 * *depthnegateIt) << ") ");
*/

      depthIt++;
      depthnegateIt++;
    }
  std::string hint;
    if (negCountb/prevFeatureD.size() > newFrame->getParameters().negativeDepthThreshold) {
      //baseLine = baseLine;
      hint = "let" + std::to_string(negCountb/prevFeatureD.size()) + "|" + std::to_string(negCountbnegate/prevFeatureD.size());
    } else if (negCountbnegate/prevFeatureD.size() > newFrame->getParameters().negativeDepthThreshold) {
      baseLine = bnegate;
      hint = "negated " + std::to_string(negCountb/prevFeatureD.size()) + "|" + std::to_string(negCountbnegate/prevFeatureD.size());
    } else {
      hint = "no Info!" + std::to_string(negCountb/prevFeatureD.size()) + "|" + std::to_string(negCountbnegate/prevFeatureD.size());
    }
    ROS_INFO_STREAM("Baseline direction : " << hint);

    /* Save Movement */
    newFrame->setBaseLineToPrevious(baseLine);
    newFrame->setScaleToPrevious(1.0);
#ifdef RATINGDATA
    newFrame->_infos.ESTIMATED_baseline = baseLine;
    newFrame->_infos.RANSAC_outsortet_features = outlier.size();
    newFrame->_infos.RANSAC_probability = 0;//TODO: Set!
    newFrame->_infos.EST_Time = ros::Time::now();
#endif

#ifdef DEBUGIMAGES
    cv::Mat image(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));
    VisualisationUtils::drawCorrespondences({&thisCorespFeaturesE, &beforeCorespFeaturesE},
                                            newFrame->getCameraModel(), image);
    std::vector<cv::Point2f> all_ft;
    newFrame->getFeatures(all_ft);
    for(unsigned int out : outlier){
      cv::circle(image, all_ft[out], 6, cv::Scalar(255,0,0), 3);
    }

    VisualisationUtils::drawMovementDebug(*newFrame, cv::Scalar(0, 0, 255), image, 0);
    cv::putText(image, hint, cv::Point(10, 40), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
    cv::imshow("mvocv-EstimatorImage", image);
    cv::waitKey(10);
#endif
  }
  //Pass through
  _prevFrame = newFrame;
  _baseLine.enqueue({newFrame->getBaseLineToPrevious(), newFrame->getRotation(), newFrame->getTimeStamp()});
  ROS_INFO_STREAM("baseLine in Queue: " << newFrame->getBaseLineToPrevious());
  return newFrame;
}
BaselineEstimator::~BaselineEstimator() {
#ifdef DEBUGIMAGES
  cv::destroyWindow("mvocv-EstimatorImage");
#endif
}

