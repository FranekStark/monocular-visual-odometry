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
                                                                           _baseLine(1) {}
Frame *BaselineEstimator::stage(Frame *newFrame) {
  if (_prevFrame == nullptr) { //If it is the first Frame
    SlidingWindow::setBaseLineToPrevious(*newFrame, cv::Vec3d(0, 0, 0));
  } else {
    /* Get CorrespondingFeatures */
    std::vector<cv::Vec3d> beforeCorespFeaturesE, thisCorespFeaturesUnrotatedE, thisCorespFeaturesE;
    SlidingWindow::getCorrespondingFeatures<cv::Vec3d>(*_prevFrame,
                                                       *newFrame,
                                                       {&beforeCorespFeaturesE, &thisCorespFeaturesE});
    /* Unroate the Features */
    auto beforeRotaton = SlidingWindow::getRotation(*_prevFrame);
    auto thisRotation = SlidingWindow::getRotation(*newFrame);
    auto diffRotation = beforeRotaton.t() * thisRotation;
    FeatureOperations::unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesUnrotatedE, diffRotation);
    /* First Guess of the Direction-BaseLine */
    std::vector<unsigned int> inlier, outlier;
    auto baseLine = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesUnrotatedE, inlier);
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
    SlidingWindow::disbandFeatureConnection(outlier, *newFrame); //TODO: don't remove, but delete the connections
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
    } else if (negCountbnegate < negCountb) {
      baseLine = bnegate;
    } else {
      ROS_WARN_STREAM("Couldn't find unambiguous solution for sign of movement." << std::endl);
    }
    /* Transform the relative BaseLine into WorldCoordinates */
    baseLine = beforeRotaton * baseLine;
    /* Save Movement */
    SlidingWindow::setBaseLineToPrevious(*newFrame, baseLine);
  }
  //Pass through
  _prevFrame = newFrame;
  _baseLine.enqueue({SlidingWindow::getBaseLineToPrevious(*newFrame), SlidingWindow::getRotation(*newFrame)});
  return newFrame;
}
BaselineEstimator::~BaselineEstimator() {

}

