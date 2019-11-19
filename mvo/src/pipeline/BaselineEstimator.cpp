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
  cv::moveWindow("mvocv-EstimatorImage", 24,551);
  cv::resizeWindow("mvocv-EstimatorImage", 1139,524);
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
    auto baseLine = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesUnrotatedE, inlier, newFrame->getParameters().bestFitProbability, threshold);
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
    newFrame->disbandFeatureConnection(outlier); //TODO: don't remove, but delete the connections
    /* Vote for the sign of the Baseline, which generates the feweset negative Gradients */
     std::vector<double> depths, depthsNegate;
    auto bnegate = -1.0 * baseLine;
    FeatureOperations::reconstructDepth(depths, thisCorespFeaturesE, beforeCorespFeaturesE, diffRotation, baseLine);
    FeatureOperations::reconstructDepth(depthsNegate,
                                        thisCorespFeaturesE,
                                        beforeCorespFeaturesE,
                                        diffRotation,
                                        bnegate);
    unsigned int negCountb = 0;
    unsigned int negCountbnegate = 0;
    assert(depths.size() == depthsNegate.size());
    auto depthIt = depths.begin();
    auto depthnegateIt = depthsNegate.begin();
    while (depthIt != depths.end()) {
      if (*depthIt < 0) { //TODO: Count here also for vanishing depths?
        negCountb++;
      }
      if (*depthnegateIt < 0) {
        negCountbnegate++;
      }
      depthIt++;
      depthnegateIt++;
    }
    if (negCountb < negCountbnegate) {
      //baseLine = baseLine;
      ROS_INFO_STREAM("Baseline positive!");
    } else if (negCountbnegate < negCountb) {
      //baseLine = bnegate;
      ROS_INFO_STREAM("Baseline negative!");
    } else {
      ROS_WARN_STREAM("Couldn't find unambiguous solution for sign of movement." << std::endl);
    }
    /* Transform the relative BaseLine into WorldCoordinates */
    baseLine = beforeRotaton * baseLine;
    /* Save Movement */
    newFrame->setBaseLineToPrevious(baseLine);
    newFrame->setScaleToPrevious(1.0);
#ifdef RATINGDATA
  newFrame->_infos.ESTIMATED_baseline = baseLine;
  newFrame->_infos.RANSAC_outsortet_features = outlier.size();
  newFrame->_infos.RANSAC_probability = 0;//TODO: Set!
#endif


#ifdef DEBUGIMAGES
    cv::Mat image(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));
    VisualisationUtils::drawCorrespondences({&thisCorespFeaturesE, &beforeCorespFeaturesE},
                                            newFrame->getCameraModel(), image);
    VisualisationUtils::drawMovementDebug(*newFrame, cv::Scalar(0,0,255), image,0);
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

