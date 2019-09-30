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
      _baseLine(1) {
  assert(numberToRefine == 3);
  //Currently only 3 available
#ifdef DEBUGIMAGES
  cv::namedWindow("RefinerImage0", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImage0", 548, 28);
  cv::resizeWindow("RefinerImage0", 622, 796);
  cv::namedWindow("RefinerImage1", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImage1", 0, 28);
  cv::resizeWindow("RefinerImage1", 600, 800);
  cv::namedWindow("RefinerImageBOTH", cv::WINDOW_NORMAL);
  cv::moveWindow("RefinerImageBOTH", 37, 880);
  cv::resizeWindow("RefinerImageBOTH", 1143, 1039);
  cv::startWindowThread();
#endif
}

Frame *Refiner::stage(Frame *newFrame) {
  if (_prePreFrame != nullptr) { //Only if enough Frames
    IterativeRefinement::RefinementDataCV data;

    data.vec0 = SlidingWindow::getBaseLineToPrevious(*newFrame);
    data.vec1 = SlidingWindow::getBaseLineToPrevious(*_preFrame);

    ROS_INFO_STREAM("BEFORE: " << std::endl
                               << "vec0: " << data.vec0 << std::endl
                               << "vec1: " << data.vec1 << std::endl);

    data.R0 = SlidingWindow::getRotation(*newFrame);
    data.R1 = SlidingWindow::getRotation(*_preFrame);
    data.R2 = SlidingWindow::getRotation(*_prePreFrame);

    std::vector<std::vector<cv::Vec3d> *> vectors{&(data.m0), &(data.m1), &(data.m2)};

    SlidingWindow::getCorrespondingFeatures(*_prePreFrame, *newFrame, vectors);

    _iterativeRefinement.refine(data);

    ROS_INFO_STREAM("After: " << std::endl
                              << "vec0: " << data.vec0 << std::endl
                              << "vec1: " << data.vec1 << std::endl);

    SlidingWindow::setBaseLineToPrevious(*newFrame, data.vec0);
    SlidingWindow::setBaseLineToPrevious(*_preFrame, data.vec1);

#ifdef DEBUGIMAGES
    cv::Mat image(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));

    VisualisationUtils::drawCorrespondences({&data.m0, &data.m1, &data.m2}, SlidingWindow::getCameraModel(*newFrame), image);
    cv::imshow("RefinerImageBOTH", image);

    cv::Mat image0(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat image1(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));
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
  if (_prePreFrame != nullptr) {
    _baseLine.enqueue({SlidingWindow::getBaseLineToPrevious(*_prePreFrame),
                       SlidingWindow::getRotation(*_prePreFrame)});
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
