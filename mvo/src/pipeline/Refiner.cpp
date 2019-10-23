//
// Created by franek on 25.09.19.
//

#include "Refiner.hpp"
#include "../operations/FeatureOperations.h"
Refiner::Refiner(PipelineStage &precursor,
                 unsigned int out_going_channel_size,
                 IterativeRefinement &iterativeRefinement,
                 unsigned int numberToRefine)
    : PipelineStage(&precursor, out_going_channel_size),
      _iterativeRefinement(iterativeRefinement),
      _preFrame(nullptr),
      _prePreFrame(nullptr),
      _baseLine1(1),
      _baseLine2(1) {
  assert(numberToRefine == 3);
  //Currently only 3 available
#ifdef DEBUGIMAGES
  cv::namedWindow("RefinerImage0", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImage0", 621, 1108);
  cv::resizeWindow("RefinerImage0", 535, 314);
  cv::namedWindow("RefinerImage1", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImage1", 0, 1103);
  cv::resizeWindow("RefinerImage1", 617, 322);
  cv::namedWindow("RefinerImageBOTH", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImageBOTH", 1, 1466);
  cv::resizeWindow("RefinerImageBOTH", 1164, 454);
  cv::startWindowThread();
#endif
}

Frame *Refiner::stage(Frame *newFrame) {
  if (_prePreFrame != nullptr) { //Only if enough Frames
    IterativeRefinement::RefinementDataCV data;

    data.vec0 = newFrame->getBaseLineToPrevious();
    if (newFrame->getParameters().usePreviousScale) { //Use Scaling from previous
      data.scale0 = _preFrame->getScaleToPrevious();
    } else { //Use Scaling from Etsimator (this is length 1)
      data.scale0 = newFrame->getScaleToPrevious();
    }

    data.vec1 = _preFrame->getBaseLineToPrevious();
    data.scale1 = _preFrame->getScaleToPrevious();

    data.R0 = newFrame->getRotation();
    data.R1 = _preFrame->getRotation();
    data.R2 = _prePreFrame->getRotation();

    std::vector<std::vector<cv::Vec3d> *> vectors{&(data.m0), &(data.m1), &(data.m2)};

    Frame::getCorrespondingFeatures(*_prePreFrame, *newFrame, vectors);
    auto funtolerance = std::pow(10.0, -1 * (newFrame->getParameters().functionTolerance));
    auto gradtolerance = std::pow(10.0, -1 * (newFrame->getParameters().gradientTolerance));
    auto paramtolerance = std::pow(10.0, -1 * (newFrame->getParameters().parameterTolerance));

    for(auto vector = vectors.begin(); vector != vectors.end(); vector++){
      FeatureOperations::normFeatures(**vector);
    }

    _iterativeRefinement.refine(data,
                                newFrame->getParameters().maxNumThreads,
                                newFrame->getParameters().maxNumIterations,
                                funtolerance,
                                gradtolerance,
                                paramtolerance,
                                newFrame->getParameters().useLossFunction,
                                newFrame->getParameters().lowestLength,
                                newFrame->getParameters().highestLength);

    newFrame->setBaseLineToPrevious(data.vec0);
    newFrame->setScaleToPrevious(data.scale0);
    _preFrame->setBaseLineToPrevious(data.vec1);
    _preFrame->setScaleToPrevious(data.scale1);

#ifdef DEBUGIMAGES
    cv::Mat image(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));

    VisualisationUtils::drawCorrespondences({&data.m0, &data.m1, &data.m2}, newFrame->getCameraModel(), image);
    cv::imshow("RefinerImageBOTH", image);

    cv::Mat image0(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat image1(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));
    //Image0
    VisualisationUtils::drawMovementDebug(*_preFrame, cv::Scalar(0, 255, 255), image1, 0);
    //Image1
    VisualisationUtils::drawMovementDebug(*newFrame, cv::Scalar(0, 255, 255), image0, 0);
    cv::imshow("RefinerImage0", image0);
    cv::imshow("RefinerImage1", image1);
    cv::waitKey(10);
#endif

  }



  //Pass the frames through
  _prePreFrame = _preFrame;
  _preFrame = newFrame;
  if (_preFrame != nullptr) {
    _baseLine1.enqueue({_preFrame->getScaleToPrevious() * _preFrame->getBaseLineToPrevious(), _preFrame->getRotation(),
                        _preFrame->getTimeStamp()});
  }

  if (_prePreFrame != nullptr) {
    _baseLine2.enqueue({_prePreFrame->getScaleToPrevious() * _prePreFrame->getBaseLineToPrevious(),
                        _prePreFrame->getRotation(), _prePreFrame->getTimeStamp()});
  }
  return _prePreFrame;
}

Refiner::~Refiner() {
#ifdef DEBUGIMAGES
  cv::destroyWindow("RefinerImage0");
  cv::destroyWindow("RefinerImage1");
  cv::destroyWindow("RefinerBOTH");
#endif
}
