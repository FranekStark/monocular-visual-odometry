#ifndef ITERATIVE_REFINEMENT_HPP
#define ITERATIVE_REFINEMENT_HPP

#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>

class IterativeRefinement {
 private:

  struct RefinementFrameEIG {
    std::vector<Eigen::Vector3d> m;
    Eigen::Matrix3d R;
    Eigen::Vector3d vec;
    double scale;
  };

  static void cvt_cv_eigen(const std::vector<cv::Vec3d> &vecaCV, std::vector<Eigen::Vector3d> &vecaEIGEN);

  static void cvt_cv_eigen(const cv::Matx33d &matCV, Eigen::Matrix3d &matEIG);

  static void cvt_cv_eigen(const cv::Vec3d &vecCV, Eigen::Vector3d &vecEIG);

  static cv::Vec3d cvt_eigen_cv(const Eigen::Vector3d &vecEIG);

  struct CostFunctionScaled {
   private:
    const Eigen::Vector3d &_m1;
    const Eigen::Vector3d &_m0;
    const Eigen::Matrix3d &_R1;
    const Eigen::Matrix3d &_R0;
    const Eigen::Vector3d _vectorOffset;
    const double _maxLength;
    const double _minlength;
    const int _params;
   public:
    CostFunctionScaled(const Eigen::Vector3d &m1,
                       const Eigen::Vector3d &m0,
                       const Eigen::Matrix3d &R1,
                       const Eigen::Matrix3d &R0,
                       const Eigen::Vector3d &vectorOffset,
                       double maxLength,
                       double minLength,
                       int params
    );

    template<typename T>
    bool operator()(T const *const *parameters, T *residuals) const;

    static void addResidualBlocks(const std::vector<Eigen::Vector3d> &m1,
                                  const std::vector<Eigen::Vector3d> &m0,
                                  const Eigen::Matrix3d &R1,
                                  const Eigen::Matrix3d &R0,
                                  ceres::LossFunction *lossFunction,
                                  std::vector<double *> &parameter_blocks,
                                  ceres::Problem &ceresProblem,
                                  Eigen::Vector3d vector_offset,
                                  double minLength,
                                  double maxLength,
                                  int params
    );
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

    static void addResidualBlocks(const std::vector<Eigen::Vector3d> &m1,
                                  const std::vector<Eigen::Vector3d> &m0,
                                  const Eigen::Matrix3d &R1,
                                  const Eigen::Matrix3d &R0,
                                  ceres::LossFunction *lossFunction,
                                  double *vectorParam, ceres::Problem &ceresProblem);

  };

  template<typename T>
  static Eigen::Matrix<T, 3, 1> baseLineTemplated(const Eigen::Matrix<T, 3, 1> &vec, const T a, const T b);

  template<typename T>
  static T scaleTemplated(const T t, double MAX_LEN, double MIN_LEN);

  static double reverseScale(const double length, double MAX_LEN, double MIN_LEN);

  struct ParametrizedBaseLine {
    template<typename T>
    bool operator()(const T *x, const T *delta, T *x_plus_delta) const;
  };

 public:

  struct RefinementFrame {
    std::vector<cv::Vec3d> m;
    cv::Matx33d R;
    cv::Vec3d vec;
    double scale;
  };

  IterativeRefinement() = default;

  ~IterativeRefinement() = default;

  void refine(std::vector<RefinementFrame> &refinementData,
              unsigned int numberToRefine,
              unsigned int numberToNote,
              int maxNumthreads,
              int maxNumIterations,
              double functionTolerance,
              double gradientTolerance,
              double parameterTolerance,
              bool useLossFunction,
              double lowestLength,
              double highestLength);

};

#endif //ITERATIVE_REFINEMENT_HPP