#ifndef ITERATIVE_REFINEMENT_HPP
#define ITERATIVE_REFINEMENT_HPP

#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include "../sliding_window/Frame.hpp"
#include "../operations/VisualisationUtils.hpp"

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_NO_AUTOMATIC_RESIZING

class IterativeRefinement {
 private:

  struct RefinementFrameEIG {
    Eigen::Matrix3d R;
    Eigen::Vector3d vec;
    double scale;
  };

  static void cvt_cv_eigen(const std::vector<cv::Vec3d> &vecaCV, std::vector<Eigen::Vector3d> &vecaEIGEN);

  static void cvt_cv_eigen(const cv::Matx33d &matCV, Eigen::Matrix3d &matEIG);

  static void cvt_cv_eigen(const cv::Vec3d &vecCV, Eigen::Vector3d &vecEIG);

  static cv::Vec3d cvt_eigen_cv(const Eigen::Vector3d &vecEIG);


  struct ScaleCostFunction{
   private:
    const Eigen::Vector3d &_m2;
    const Eigen::Vector3d &_m1;
    const Eigen::Vector3d &_m0;
    const Eigen::Matrix3d &_R2;
    const Eigen::Matrix3d &_R1;
    const Eigen::Matrix3d &_R0;
    const double _maxLength;
    const double _minlength;
   public:
    ScaleCostFunction(const Eigen::Vector3d &m2,
                      const Eigen::Vector3d &m1,
                      const Eigen::Vector3d &m0,
                      const Eigen::Matrix3d &r2,
                      const Eigen::Matrix3d &r1,
                      const Eigen::Matrix3d &r0,
                      const double max_length,
                      const double min_length);
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,const T *scale1, const T *vec1,T *residuals) const;
  };

  struct CostFunctionBase {
   private:
    const Eigen::Vector3d &_m1;
    const Eigen::Vector3d &_m0;
    const Eigen::Matrix3d &_R1;
    const Eigen::Matrix3d &_R0;
    const Eigen::Vector3d _vectOffset;
   protected:
    const double _maxLength;
    const double _minlength;
    template<typename T>
    T cost(const Eigen::Matrix<T, 3, 1> &baseLine) const;
   public:
    CostFunctionBase(const Eigen::Vector3d &m1,
                     const Eigen::Vector3d &m0,
                     const Eigen::Matrix3d &R1,
                     const Eigen::Matrix3d &R0,
                     double maxLength,
                     double minLength, const Eigen::Vector3d &vectOffset);
    virtual ~CostFunctionBase() = default;
  };

  struct CostFunction1 : CostFunctionBase {
    CostFunction1(const Eigen::Vector3d &m1,
                  const Eigen::Vector3d &m0,
                  const Eigen::Matrix3d &r1,
                  const Eigen::Matrix3d &r0,
                  double max_length,
                  double min_length,
                  const Eigen::Vector3d &vect_offset);
    virtual ~CostFunction1() = default;
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,T *residuals) const;
  };

  struct CostFunction2 : CostFunctionBase {
    CostFunction2(const Eigen::Vector3d &m1,
                  const Eigen::Vector3d &m0,
                  const Eigen::Matrix3d &r1,
                  const Eigen::Matrix3d &r0,
                  double max_length,
                  double min_length,
                  const Eigen::Vector3d &vect_offset);
    virtual ~CostFunction2() = default;
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,const T *scale1, const T *vec1,T *residuals) const;
  };

  struct CostFunction3 : CostFunctionBase {
    CostFunction3(const Eigen::Vector3d &m1,
                  const Eigen::Vector3d &m0,
                  const Eigen::Matrix3d &r1,
                  const Eigen::Matrix3d &r0,
                  double max_length,
                  double min_length,
                  const Eigen::Vector3d &vect_offset);
    virtual ~CostFunction3() = default;
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,const T *scale1, const T *vec1, const T *scale2, const T *vec2,T *residuals) const;
  };


  struct CostFunction4 : CostFunctionBase {
    CostFunction4(const Eigen::Vector3d &m1,
                  const Eigen::Vector3d &m0,
                  const Eigen::Matrix3d &r1,
                  const Eigen::Matrix3d &r0,
                  double max_length,
                  double min_length,
                  const Eigen::Vector3d &vect_offset);
    virtual ~CostFunction4() = default;
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,const T *scale1, const T *vec1, const T *scale2, const T *vec2, const T *scale3, const T *vec3,T *residuals) const;
  };

  struct CostFunction5 : CostFunctionBase {
    CostFunction5(const Eigen::Vector3d &m1,
                  const Eigen::Vector3d &m0,
                  const Eigen::Matrix3d &r1,
                  const Eigen::Matrix3d &r0,
                  double max_length,
                  double min_length,
                  const Eigen::Vector3d &vect_offset);
    virtual ~CostFunction5() = default;
    template<typename T>
    bool operator()(const T *scale0, const T *vec0,const T *scale1, const T *vec1, const T *scale2, const T *vec2, const T *scale3, const T *vec3,const T *scale4, const T *vec4,T *residuals) const;
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


  static void addResidualBlocks(const std::vector<Eigen::Vector3d> & features1, const std::vector<Eigen::Vector3d> & features0,
                                const Eigen::Matrix3d & R1, const Eigen::Matrix3d & R0,
                                std::vector<double *> parameter_blocks, const Eigen::Vector3d & vect_offset,
                                ceres::LossFunction * loss_fun, ceres::Problem & ceres_problem, double highest_len, double lowest_len);



 public:
#ifdef DEBUGIMAGES
  cv::Mat _debugImage;
#endif

  struct RefinementFrame {
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
              double highestLength,
              const Frame &nowFrame);

};

#endif //ITERATIVE_REFINEMENT_HPP