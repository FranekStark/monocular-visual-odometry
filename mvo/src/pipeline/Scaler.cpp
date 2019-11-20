//
// Created by franek on 15.11.19.
//

#include "Scaler.hpp"
Scaler::Scaler(PipelineStage &precursor, unsigned int out_going_channel_size)
    : PipelineStage(&precursor, out_going_channel_size), _prevFrame(nullptr), _keptFrame(nullptr) {}

Scaler::~Scaler() {
  delete _keptFrame;
}

Frame *Scaler::stage(Frame *newFrame) {

  if(_keptFrame != nullptr && newFrame != nullptr && _prevFrame != nullptr) {
    const double & n1 = _keptFrame->getScaleToPrevious();
    const cv::Vec3d & u1 = _keptFrame->getBaseLineToPrevious();
    cv::Vec3d u1norm = u1 / cv::norm(u1);
    std::vector<cv::Vec3d> v2, v1, v0;
    Frame::getCorrespondingFeatures<cv::Vec3d>(*_prevFrame, *newFrame, {&v0,&v1,&v2});

    auto m2 = v2.begin();
    auto m1 = v1.begin();
    auto m0 = v0.begin();

    for(;m0 != v0.end(); m0++, m1++, m2++){
      double counting = n1 * std::sqrt(1.0 - std::pow(m2->dot(u1norm),2)) * std::sqrt(1.0 - std::pow(m1->dot(*m0),2));
      double naming = std::sqrt(1.0 - std::pow(m2->dot(*m1),2)) * std::sqrt(1.0 - std::pow(u1norm.dot(*m0),2));
      double n0 = counting / naming;
      ROS_INFO_STREAM("n0: " << n0);
    }

  }

  _prevFrame = _keptFrame;
  _keptFrame =  newFrame;
  return _prevFrame ;
}
