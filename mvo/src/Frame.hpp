#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv2/core.hpp>
#include <vector>
#include "Feature.hpp"

enum FrameType
{
    TEMP, PERSIST
};

struct Frame
{
    std::vector<Feature> _features;
    cv::Mat _image;

    cv::Vec3d _position;
    cv::Matx33d _rotation;

    Frame * _preFrame;

    FrameType _type;
};

#endif //FRAME_HPP