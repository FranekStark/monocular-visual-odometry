#ifndef EPIPOLAR_GEOMETRY_HPP
#define EPIPOLAR_GEOMETRY_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <boost/math/special_functions/binomial.hpp>

#include <random>

class EpipolarGeometry {
 private:
  const double PI;
  const double THRESHOLD; //Ab wann sind Punkte Outlier
  const double Ps; //Gewünschte Wahrschenilchkeit für best Fit

  /*Random Values */
  std::random_device _randomDevice;
  std::mt19937 _randomGenerator;

  cv::Vec3d calculateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt);

  unsigned int estimateNumberOfIteration(unsigned int N, double inlierProbability, unsigned int s);

  unsigned int reEstimateNumberOfIteration(unsigned int N, unsigned int nInlier, unsigned int s);

 public:
  EpipolarGeometry();

  ~EpipolarGeometry() = default;

  cv::Vec3d estimateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt,
                             std::vector<unsigned int> &inlierIndexes);
};

#endif //EPIPOLAR_GEOMETRY_HPP