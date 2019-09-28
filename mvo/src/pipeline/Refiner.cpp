//
// Created by franek on 25.09.19.
//

#include "Refiner.hpp"
Refiner::Refiner(PipelineStage &precursor,
                 unsigned int out_going_channel_size,
                 IterativeRefinement &iterativeRefinement,
                 unsigned int numberToRefine)
    : PipelineStage(&precursor, out_going_channel_size),
      _iterativeRefinement(iterativeRefinement),
      _ringBuffer(numberToRefine - 1),
      _baseLine(1) {
  assert(numberToRefine == 3);
  //Currently only 3 available
#ifdef DEBUGIMAGES
  cv::namedWindow("RefinerImage0", cv::WINDOW_NORMAL);
  cv::namedWindow("RefinerImage1", cv::WINDOW_NORMAL);
  cv::namedWindow("RefinerImageBOTH", cv::WINDOW_NORMAL);
  cv::startWindowThread();
#endif
}

Frame *Refiner::stage(Frame *newFrame) {
  if (_ringBuffer.full()) { //Only if enough Frames
    IterativeRefinement::RefinementDataCV data;

    cv::Vec3d vec0 = SlidingWindow::getBaseLineToPrevious(*newFrame);
    cv::Vec3d vec1 = SlidingWindow::getBaseLineToPrevious(*_ringBuffer[1]);

    ROS_INFO_STREAM("BEFORE: " << std::endl
                               << "vec0: " << vec0 << std::endl
                               << "vec1: " << vec1 << std::endl);

    double n0 = cv::norm(vec0);
    double n1 = cv::norm(vec1);
    cv::Vec3d u0 = (vec0) / n0;
    cv::Vec3d u1 = (vec1) / n1;

    data.vec0 = u0;
    data.vec1 = u1;

    data.R0 = SlidingWindow::getRotation(*newFrame);
    data.R1 = SlidingWindow::getRotation(*_ringBuffer[1]);
    data.R2 = SlidingWindow::getRotation(*_ringBuffer[0]);

    std::vector<std::vector<cv::Vec3d> *> vectors{&(data.m0), &(data.m1), &(data.m2)};

    SlidingWindow::getCorrespondingFeatures(*_ringBuffer[0], *newFrame, vectors);

    ROS_INFO_STREAM("After: " << std::endl
                              << "vec0: " << data.vec0 << std::endl
                              << "vec1: " << data.vec1 << std::endl);

    SlidingWindow::setBaseLineToPrevious(*newFrame, vec0);
    SlidingWindow::setBaseLineToPrevious(*_ringBuffer[1], vec1);

  }

#ifdef DEBUGIMAGES
  cv::Mat image(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));
  VisualisationUtils::drawCorrespondences(*_ringBuffer[0], *newFrame, image);
  cv::imshow("RefinerImageBOTH", )
#endif

  //Pass the frames through
  _ringBuffer.pop();
  _ringBuffer.push(newFrame);

#ifdef DEBUGIMAGES
  if (_ringBuffer.full()) {
    cv::Mat image0(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat image1(SlidingWindow::getImage(*newFrame).size(), CV_8UC3, cv::Scalar(100, 100, 100));
    //Image0
    VisualisationUtils::drawMovementDebug(*_ringBuffer[1], cv::Scalar(0,255,255), image0, 0);
    //Image1
    VisualisationUtils::drawMovementDebug(*_ringBuffer[0], cv::Scalar(0,255,255), image0, 0);
  }
#endif


  if (_ringBuffer[0] != nullptr) {
    _baseLine.enqueue({SlidingWindow::getBaseLineToPrevious(*_ringBuffer[0]),
                       SlidingWindow::getRotation(*_ringBuffer[0])});
  }
  return _ringBuffer[0];
}

Refiner::~Refiner() {
#ifdef DEBUGIMAGES
  cv::destroyWindow("RefinerImage0");
  cv::destroyWindow("RefinerImage1");
  cv::destroyWindow("RefinerBOTH");
#endif
}
