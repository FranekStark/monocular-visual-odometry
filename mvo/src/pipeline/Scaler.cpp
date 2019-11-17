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

  if (_keptFrame != nullptr && newFrame != nullptr && _prevFrame != nullptr) {

    double sumN0 = 0.0;
    unsigned int numN0 = 0;

    const double &n1 = _keptFrame->getScaleToPrevious();
    const cv::Vec3d &u1 = _keptFrame->getBaseLineToPrevious();
    std::vector<cv::Vec3d> v2, v1, v0;
    Frame::getCorrespondingFeatures<cv::Vec3d>(*_prevFrame, *newFrame, {&v0, &v1, &v2});

    if (v0.size() == 0) {
      ROS_ERROR_STREAM("Cannot obtain Scale, cause no Featurescorrespondence through 3-Window!");
      }else{

              auto m2 = v2.begin();
              auto m1 = v1.begin();
              auto m0 = v0.begin();

              for (; m0 != v0.end(); m0++, m1++, m2++) {
                cv::Vec3d m0norm = *m0 / cv::norm(*m0);
                cv::Vec3d m1norm = *m1 / cv::norm(*m1);
                cv::Vec3d m2norm = *m2 / cv::norm(*m2);

                double counting = n1 * std::sqrt(1.0 - std::pow(m2norm.dot(u1), 2))
                    * std::sqrt(1.0 - std::pow(m1norm.dot(m0norm), 2));
                double naming =
                    std::sqrt(1.0 - std::pow(m2norm.dot(m1norm), 2)) * std::sqrt(1.0 - std::pow(u1.dot(m0norm), 2));
                double n0 = counting / naming;
                sumN0 += n0;
                numN0++;
                ROS_INFO_STREAM("scaling cal n0 " << n0);
              }

              double n0 = sumN0 / numN0;

              const auto &params = newFrame->getParameters();
              if (n0 > params.highestLength) {
                ROS_ERROR_STREAM("Highest Length to low:" << n0);
                newFrame->setScaleToPrevious(params.highestLength);
              } else if (n0 < params.lowestLength) {
                ROS_ERROR_STREAM("Lowest Length to High:" << n0);
                newFrame->setScaleToPrevious(params.lowestLength);
              } else {
                newFrame->setScaleToPrevious(n0);
                ROS_INFO_STREAM("Scaled: " << n0);
              }
            }

          }

          _prevFrame = _keptFrame;
          _keptFrame = newFrame;
          return _prevFrame;
        }
