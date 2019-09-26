#include <opencv2/core.hpp>
#include "../sliding_window/SlidingWindow.hpp"
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>

class IterativeRefinement {
 private:

  double THRESHOLD = 0.0001;



  struct RefinementDataEIG {
    std::vector<Eigen::Vector3d> m2;
    std::vector<Eigen::Vector3d> m1;
    std::vector<Eigen::Vector3d> m0;
    Eigen::Matrix3d R2;
    Eigen::Matrix3d R1;
    Eigen::Matrix3d R0;
    Eigen::Vector3d vec0;
    Eigen::Vector3d vec1;
  };

  void cvt_cv_eigen(const std::vector<cv::Vec3d> &vecaCV, std::vector<Eigen::Vector3d> &vecaEIGEN);

  void cvt_cv_eigen(const cv::Matx33d &matCV, Eigen::Matrix3d &matEIG);

  void cvt_cv_eigen(const cv::Vec3d &vecCV, Eigen::Vector3d &vecEIG);

  cv::Vec3d cvt_eigen_cv(const Eigen::Vector3d &vecEIG);

  struct CostFunctionScaled {
   private:
    const Eigen::Vector3d &_m2;
    const Eigen::Vector3d &_m1;
    const Eigen::Vector3d &_m0;
    const Eigen::Matrix3d &_R2;
    const Eigen::Matrix3d &_R1;
    const Eigen::Matrix3d &_R0;
   public:
    CostFunctionScaled(const Eigen::Vector3d &m2,
                       const Eigen::Vector3d &m1,
                       const Eigen::Vector3d &m0,
                       const Eigen::Matrix3d &R2,
                       const Eigen::Matrix3d &R1,
                       const Eigen::Matrix3d &R0);

    template<typename T>
    bool operator()(const T *vec0, const T *vec1, const T *scale0, const T *scale1, T *residuals) const;
  };

  struct CostFunction {
   private:
    const Eigen::Vector3d &_m1;
    const Eigen::Vector3d &_m0;
    const Eigen::Matrix3d &_R1;
    const Eigen::Matrix3d &_R0;
   public:
    CostFunction(
        const Eigen::Vector3d &m1,
        const Eigen::Vector3d &m0,
        const Eigen::Matrix3d &R1,
        const Eigen::Matrix3d &R0);

    template<typename T>
    bool operator()(const T *vec, T *residuals) const;

  };

  static constexpr double LOW_VALUE = 0.25;
  static constexpr double HIGH_VALUE = 2;

  template<typename T>
  static Eigen::Matrix<T, 3, 1> baseLineTemplated(const Eigen::Matrix<T, 3, 1> &vec, const T a, const T b);

  template<typename T>
  static T scaleTemplated(const T t);

  static double reverseScale(const double length);

  struct ParametrizedBaseLine {
    template<typename T>
    bool operator()(const T *x, const T *delta, T *x_plus_delta) const;
  };

 public:

  struct RefinementDataCV {
    std::vector<cv::Vec3d> m2;
    std::vector<cv::Vec3d> m1;
    std::vector<cv::Vec3d> m0;
    cv::Matx33d R2;
    cv::Matx33d R1;
    cv::Matx33d R0;
    cv::Vec3d vec0;
    cv::Vec3d vec1;
  };

  IterativeRefinement();

  ~IterativeRefinement();

  void refine(RefinementDataCV &refinementData);

};
