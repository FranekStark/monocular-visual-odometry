#ifndef EPIPOLAR_GEOMETRY_HPP
#define EPIPOLAR_GEOMETRY_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <boost/math/special_functions/binomial.hpp>
#include <cmath>

#include <random>

class EpipolarGeometry {
 private:
  const double PI;

  /*Random Values */
  std::random_device _randomDevice;
  std::mt19937 _randomGenerator;

  cv::Vec3d calculateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt);

  unsigned int estimateNumberOfIteration(unsigned int N, double inlierProbability, unsigned int s, double ps);

  unsigned int reEstimateNumberOfIteration(unsigned int N, unsigned int nInlier, unsigned int s, double ps);

 public:
  EpipolarGeometry();

  ~EpipolarGeometry() = default;
/**
 *
 *
 * @param mhi
 * @param mt
 * @param inlierIndexes
 * @param ps
 * @param threshold
 * @return
 */
  cv::Vec3d estimateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt,
                             std::vector<unsigned int> &inlierIndexes, double ps, double threshold);
};

#endif //EPIPOLAR_GEOMETRY_HPP