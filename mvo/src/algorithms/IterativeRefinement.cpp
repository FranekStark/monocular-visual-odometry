#include "IterativeRefinement.hpp"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>
#include <exception>
#include <typeinfo>
#include <string>
#include <sstream>
#include <iostream>

void IterativeRefinement::refine(RefinementDataCV &refinementData,
                                 int maxNumthreads,
                                 int maxNumIterations,
                                 double functionTolerance,
                                 double gradientTolerance,
                                 double parameterTolerance,
                                 bool useLossFunction,
                                 double lowestLength,
                                 double highestLength) {

  RefinementDataEIG dataEIG;
  cvt_cv_eigen(refinementData.m0, dataEIG.m0);
  cvt_cv_eigen(refinementData.m1, dataEIG.m1);
  cvt_cv_eigen(refinementData.m2, dataEIG.m2);
  cvt_cv_eigen(refinementData.R0, dataEIG.R0);
  cvt_cv_eigen(refinementData.R1, dataEIG.R1);
  cvt_cv_eigen(refinementData.R2, dataEIG.R2);
  cvt_cv_eigen(refinementData.vec0, dataEIG.vec0);
  cvt_cv_eigen(refinementData.vec1, dataEIG.vec1);

  double scale0[1] = {reverseScale(refinementData.scale0, highestLength, lowestLength)};
  double scale1[1] = {reverseScale(refinementData.scale1, highestLength, lowestLength)};

  double vec0[3] = {dataEIG.vec0(0), dataEIG.vec0(1), dataEIG.vec0(2)};
  double vec1[3] = {dataEIG.vec1(0), dataEIG.vec1(1), dataEIG.vec1(2)};

  ROS_INFO_STREAM("Before: ");
  ROS_INFO_STREAM("n0 * u0: " << refinementData.scale0 << " * [" << vec0[0] << "," << vec0[1] << "," << vec0[2] << "]");
  ROS_INFO_STREAM("n1 * u1: " << refinementData.scale1 << " * [" << vec1[0] << "," << vec1[1] << "," << vec1[2] << "]");

  ceres::Problem ceres_problem;
  ceres::Solver::Options ceres_solver_options;
  ceres_solver_options.minimizer_type = ceres::TRUST_REGION;
  ceres_solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  ceres_solver_options.linear_solver_type = ceres::DENSE_QR;
  ceres_solver_options.max_num_iterations = maxNumIterations;
  ceres_solver_options.num_threads = maxNumthreads;
  ceres_solver_options.function_tolerance = functionTolerance;
  ceres_solver_options.gradient_tolerance = gradientTolerance;
  ceres_solver_options.parameter_tolerance = parameterTolerance;
  //ceres_solver_options.check_gradients = true; ///DEBUG!
  //ceres_solver_options.minimizer_progress_to_stdout = true; ///DEBUG!


  for (unsigned int i = 0; i < refinementData.m0.size(); i++) {
    ceres::CostFunction *cost_functiom20 = new ceres::AutoDiffCostFunction<CostFunctionScaled, 1, 3, 3, 1, 1>(
        new CostFunctionScaled(dataEIG.m2[i],
                               dataEIG.m1[i],
                               dataEIG.m0[i],
                               dataEIG.R2,
                               dataEIG.R1,
                               dataEIG.R0,
                               highestLength,
                               lowestLength)
    );
    ceres::CostFunction *cost_functiom21 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
        new CostFunction(dataEIG.m2[i], dataEIG.m1[i], dataEIG.R2, dataEIG.R1)
    );
    ceres::CostFunction *cost_functiom10 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
        new CostFunction(dataEIG.m1[i], dataEIG.m0[i], dataEIG.R1, dataEIG.R0)
    );

    ceres::LossFunction *loss_function20 = nullptr;
    ceres::LossFunction *loss_function21 = nullptr;
    ceres::LossFunction *loss_function10 = nullptr;
    if (useLossFunction) {
      loss_function20 = new ceres::CauchyLoss(0.5);
      loss_function21 = new ceres::CauchyLoss(0.5);
      loss_function10 = new ceres::CauchyLoss(0.5);
    }
    ceres_problem.AddResidualBlock(cost_functiom20, loss_function20, vec0, vec1, scale0, scale1);
    ceres_problem.AddResidualBlock(cost_functiom21, loss_function21, vec1);
    ceres_problem.AddResidualBlock(cost_functiom10, loss_function10, vec0);
  }

  ceres::LocalParameterization
      *local_parametrizationVec0 = new ceres::AutoDiffLocalParameterization<ParametrizedBaseLine, 3, 2>;
  ceres::LocalParameterization
      *local_parametrizationVec1 = new ceres::AutoDiffLocalParameterization<ParametrizedBaseLine, 3, 2>;
  ceres_problem.SetParameterization(vec0, local_parametrizationVec0);
  ceres_problem.SetParameterization(vec1, local_parametrizationVec1);

  //ceres_solver_options.minimizer_progress_to_stdout = true;
  //ceres_solver_options.check_gradients = true;
  // std::vector<int> it_to_dump;
  // it_to_dump.push_back(0);
  // it_to_dump.push_back(1);
  // it_to_dump.push_back(2);
  // it_to_dump.push_back(3);
  // ceres_solver_options.trust_region_minimizer_iterations_to_dump = it_to_dump;


  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options, &ceres_problem, &ceres_summary);

  if(ceres_summary.termination_type == ceres::TerminationType::FAILURE){
    throw std::runtime_error("CERES ERROR!");
  }

  auto n0 = scaleTemplated<double>(scale0[0], highestLength, lowestLength);  // T0
  auto n1 = scaleTemplated<double>(scale1[0], highestLength, lowestLength);  // T1

  ROS_INFO_STREAM("After: ");
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * [" << vec0[0] << "," << vec0[1] << "," << vec0[2] << "]");
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * [" << vec1[0] << "," << vec1[1] << "," << vec1[2] << "]");

  cv::Vec3d u0 = cvt_eigen_cv(Eigen::Vector3d(vec0[0], vec0[1], vec0[2]));
  cv::Vec3d u1 = cvt_eigen_cv(Eigen::Vector3d(vec1[0], vec1[1], vec1[2]));

  ROS_WARN_STREAM_COND((cv::norm(u0) != 1.0), "Vector0 isn't a unitvector, len: " << cv::norm(u0));
  ROS_WARN_STREAM_COND((cv::norm(u1) != 1.0), "Vector1 isn't a unitvector, len:" << cv::norm(u1));

  ROS_INFO_STREAM(ceres_summary.FullReport());

  refinementData.vec0 = u0;
  refinementData.vec1 = u1;
  refinementData.scale0 = n0;
  refinementData.scale1 = n1;
}

IterativeRefinement::CostFunctionScaled::CostFunctionScaled(const Eigen::Vector3d &m2,
                                                            const Eigen::Vector3d &m1,
                                                            const Eigen::Vector3d &m0,
                                                            const Eigen::Matrix3d &R2,
                                                            const Eigen::Matrix3d &R1,
                                                            const Eigen::Matrix3d &R0,
                                                            double maxLength,
                                                            double minLength) :
    _m2(m2),
    _m1(m1),
    _m0(m0),
    _R2(R2),
    _R1(R1),
    _R0(R0),
    _maxLength(maxLength),
    _minlength(minLength) {}

template<typename T>
bool IterativeRefinement::CostFunctionScaled::operator()(const T *vec0, const T *vec1, const T *scale0, const T *scale1,
                                                         T *residuals) const {
  T n0 = scaleTemplated(scale0[0], _maxLength, _minlength);
  T n1 = scaleTemplated(scale1[0], _maxLength, _minlength);

  Eigen::Matrix<T, 3, 1> u0;
  u0 << vec0[0], vec0[1], vec0[2];
  Eigen::Matrix<T, 3, 1> u1;
  u1 << vec1[0], vec1[1], vec1[2];
  Eigen::Matrix<T, 3, 1> u01 = n1 * u1 + n0 * u0;
  u01.normalize();
  T cost = ((_m2.template cast<T>()).dot(
      (_R2).transpose().template cast<T>() * u01.cross((_R0).template cast<T>() * (_m0).template cast<T>())));
  residuals[0] = cost;

  return true;
}

IterativeRefinement::CostFunction::CostFunction(const Eigen::Vector3d &m1,
                                                const Eigen::Vector3d &m0,
                                                const Eigen::Matrix3d &R1,
                                                const Eigen::Matrix3d &R0) :
    _m1(m1),
    _m0(m0),
    _R1(R1),
    _R0(R0) {}

template<typename T>
bool IterativeRefinement::CostFunction::operator()(const T *vec, T *residuals) const {

  Eigen::Matrix<T, 3, 1> u;
  u << vec[0], vec[1], vec[2];

  T cost = ((_m1.template cast<T>()).dot(
      _R1.transpose().template cast<T>() * u.cross(_R0.template cast<T>() * _m0.template cast<T>())));
  residuals[0] = cost;
  return true;
}

template<typename T>
Eigen::Matrix<T, 3, 1> IterativeRefinement::baseLineTemplated(const Eigen::Matrix<T, 3, 1> &vec, const T a, const T b) {
  Eigen::Matrix<T, 3, 3> A;
  A << (T(1.0) - a * a) / (T(1.0) + a * a), (T(-2.0) * a) / (T(1.0) + a * a), T(0),
      (T(2.0) * a) / (T(1.0) + a * a), (T(1.0) - a * a) / (T(1.0) + a * a), T(0),
      T(0.0), T(0.0), T(1.0);

  Eigen::Matrix<T, 3, 3> B;
  B << (T(1.0) - b * b) / (T(1.0) + b * b), T(0), (T(2.0) * b) / (T(1.0) + b * b),
      T(0.0), T(1.0), T(0.0),
      (T(-2.0) * b) / (T(1.0) + b * b), T(0), (T(1.0) - b * b) / (T(1.0) + b * b);

  return (B * A * vec.template cast<T>());

}

template<typename T>
T IterativeRefinement::scaleTemplated(T t, double MAX_LEN, double MIN_LEN) {
  T result;
  auto exp = ceres::exp(-1.0 * t);
  if(ceres::IsInfinite(exp)){ //In Case, that this term gets infinite. The whole function is instable for derivations i guess. (It results in "nan" in ceres. Therefore we have to cattch need to know, that 1/Inf ~= 0. So we have to return MIN_VALUE.
    result = T(MIN_LEN);
  }else {
    result = MIN_LEN + ((MAX_LEN - MIN_LEN) / (1.0 + exp));
  }
  return result;
}

void IterativeRefinement::cvt_cv_eigen(const std::vector<cv::Vec3d> &vecaCV, std::vector<Eigen::Vector3d> &vecaEIGEN) {
  vecaEIGEN.resize(vecaCV.size());
  auto cvIT = vecaCV.begin();
  auto eiIT = vecaEIGEN.begin();
  for (; cvIT != vecaCV.end(); cvIT++, eiIT++) {
    *eiIT << (*cvIT)(0), (*cvIT)(1), (*cvIT)(2);
  }
}

void IterativeRefinement::cvt_cv_eigen(const cv::Matx33d &matCV, Eigen::Matrix3d &matEIG) {
  matEIG << matCV(0, 0), matCV(0, 1), matCV(0, 2),
      matCV(1, 0), matCV(1, 1), matCV(1, 2),
      matCV(2, 0), matCV(2, 1), matCV(2, 2);
}

void IterativeRefinement::cvt_cv_eigen(const cv::Vec3d &vecCV, Eigen::Vector3d &vecEIG) {
  vecEIG << vecCV(0), vecCV(1), vecCV(2);
}

cv::Vec3d IterativeRefinement::cvt_eigen_cv(const Eigen::Vector3d &vecEIG) {
  return cv::Vec3d(vecEIG(0), vecEIG(1), vecEIG(2));
}

template<typename T>
bool IterativeRefinement::ParametrizedBaseLine::operator()(const T *x, const T *delta, T *x_plus_delta) const {
  Eigen::Matrix<T, 3, 1> vec;
  vec << x[0], x[1], x[2];

  auto vecRefine = baseLineTemplated(vec, delta[0], delta[1]);

  x_plus_delta[0] = vecRefine[0];
  x_plus_delta[1] = vecRefine[1];
  x_plus_delta[2] = vecRefine[2];

  return true;
}

double IterativeRefinement::reverseScale(const double length, double MAX_LEN, double MIN_LEN) {
  double t = 1;
  if (length <=
      MIN_LEN) //Catch the Cases in which the SCaling is to low or high, cause it is Mathematical impossible to calc t
  {
    t = reverseScale(MIN_LEN + (length * std::numeric_limits<double>::epsilon()) +  std::numeric_limits<double>::epsilon(), MAX_LEN, MIN_LEN);
    ROS_WARN_STREAM("Lower bound of scaling to high: " << length);
  } else if (length >= MAX_LEN) {
    t = reverseScale(MAX_LEN - (length * std::numeric_limits<double>::epsilon() +  std::numeric_limits<double>::epsilon()), MAX_LEN, MIN_LEN);
    ROS_WARN_STREAM("Upper bound of scaling to low: " << length);
  } else {
    t = -1.0 * std::log((MAX_LEN - length) / (length - MIN_LEN));
  }
  return t;
}