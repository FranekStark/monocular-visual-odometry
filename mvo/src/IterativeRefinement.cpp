#include "IterativeRefinement.hpp"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>



// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(SlidingWindow &slidingWindow) : _slidingWindow(slidingWindow) {
}

IterativeRefinement::~IterativeRefinement() {
}

void IterativeRefinement::refine(unsigned int firstFrame, unsigned int secondFrame) {

  _slidingWindow.exportMatlabData();
  ROS_INFO_STREAM("BEFORE: " << std::endl
                             << "st0: " << _slidingWindow.getPosition(0) << std::endl
                             << "st1: " << _slidingWindow.getPosition(1) << std::endl
                             << "st2: " << _slidingWindow.getPosition(2) << std::endl);

  assert((firstFrame - secondFrame) == 2);  // TODO: currently only WindowSize 3 (diff = 2) available
  RefinementDataCV data;

  cv::Vec3d &st0 = _slidingWindow.getPosition(secondFrame);
  cv::Vec3d &st1 = _slidingWindow.getPosition(secondFrame + 1);
  cv::Vec3d &st2 = _slidingWindow.getPosition(firstFrame);

  double n0 = cv::norm(st0 - st1);
  double n1 = cv::norm(st1 - st2);
  ROS_INFO_STREAM("norm n0: " << n0 << std::endl);
  ROS_INFO_STREAM("norm n1: " << n1 << std::endl);
  cv::Vec3d u0 = (st0 - st1) / n0;
  cv::Vec3d u1 = (st1 - st2) / n1;

  ROS_INFO_STREAM("Before: " << std::endl);
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * " << u0 << std::endl);
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * " << u1 << std::endl);

  data.vec0 = u0;
  data.vec1 = u1;

  data.R0 = _slidingWindow.getRotation(secondFrame);
  data.R1 = _slidingWindow.getRotation(secondFrame + 1);
  data.R2 = _slidingWindow.getRotation(firstFrame);

  std::vector<std::vector<cv::Vec3d> *> vectors{&(data.m0), &(data.m1), &(data.m2)};

  _slidingWindow.getCorrespondingFeatures(firstFrame, secondFrame, vectors);

  RefinementDataEIG dataEIG;
  cvt_cv_eigen(data.m0, dataEIG.m0);
  cvt_cv_eigen(data.m1, dataEIG.m1);
  cvt_cv_eigen(data.m2, dataEIG.m2);
  cvt_cv_eigen(data.R0, dataEIG.R0);
  cvt_cv_eigen(data.R1, dataEIG.R1);
  cvt_cv_eigen(data.R2, dataEIG.R2);
  cvt_cv_eigen(data.vec0, dataEIG.vec0);
  cvt_cv_eigen(data.vec1, dataEIG.vec1);

  double scale0[1] = {reverseScale(n0)};
  double scale1[1] = {reverseScale(n1)};

  double vec0[3] = {dataEIG.vec0(0), dataEIG.vec0(1), dataEIG.vec0(2)};
  double vec1[3] = {dataEIG.vec1(0), dataEIG.vec1(1), dataEIG.vec1(2)};

  ceres::Problem ceres_problem;
  ceres::Solver::Options ceres_solver_options;
  ceres_solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  ceres_solver_options.linear_solver_type = ceres::DENSE_QR;
  ceres_solver_options.max_num_iterations = 500;
  ceres_solver_options.num_threads = 8;
  ceres_solver_options.function_tolerance = 1e-10;
  //ceres_solver_options.check_gradients = true; ///DEBUG!


  for (unsigned int i = 0; i < data.m0.size(); i++) {
    ceres::CostFunction *cost_functiom20 = new ceres::AutoDiffCostFunction<CostFunctionScaled, 1, 3, 3, 1, 1>(
        new CostFunctionScaled(dataEIG.m2[i], dataEIG.m1[i], dataEIG.m0[i], dataEIG.R2, dataEIG.R1, dataEIG.R0)
    );
    ceres::CostFunction *cost_functiom21 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
        new CostFunction(dataEIG.m2[i], dataEIG.m1[i], dataEIG.R2, dataEIG.R1)
    );
    ceres::CostFunction *cost_functiom10 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
        new CostFunction(dataEIG.m1[i], dataEIG.m0[i], dataEIG.R1, dataEIG.R0)
    );

    // ceres::LossFunction *loss_function20 = new ceres::CauchyLoss(0.5);
    // ceres::LossFunction *loss_function21 = new ceres::CauchyLoss(0.5);
    // ceres::LossFunction *loss_function10 = new ceres::CauchyLoss(0.5);
    ceres_problem.AddResidualBlock(cost_functiom20, NULL, vec0, vec1, scale0, scale1);
    ceres_problem.AddResidualBlock(cost_functiom21, NULL, vec1);
    ceres_problem.AddResidualBlock(cost_functiom10, NULL, vec0);
  }

  ceres::LocalParameterization
      *local_parametrization = new ceres::AutoDiffLocalParameterization<ParametrizedBaseLine, 3, 2>;
  ceres_problem.SetParameterization(vec0, local_parametrization);
  ceres_problem.SetParameterization(vec1, local_parametrization);

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

  ROS_INFO_STREAM(ceres_summary.FullReport() << std::endl);

  n0 = scaleTemplated<double>(scale0[0]);  // T0
  n1 = scaleTemplated<double>(scale1[0]);  // T1

  u0 = cvt_eigen_cv(Eigen::Vector3d(vec0[0], vec0[1], vec0[2]));
  u1 = cvt_eigen_cv(Eigen::Vector3d(vec1[0], vec1[1], vec1[2]));

  ROS_INFO_STREAM("After: " << std::endl);
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * " << u0 << std::endl);
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * " << u1 << std::endl);

  st1 = (n1 * u1) + st2;
  st0 = (n0 * u0) + st1;

  ROS_INFO_STREAM("After: " << std::endl
                            << "st0: " << _slidingWindow.getPosition(0) << std::endl
                            << "st1: " << _slidingWindow.getPosition(1) << std::endl
                            << "st2: " << _slidingWindow.getPosition(2) << std::endl);
}

IterativeRefinement::CostFunctionScaled::CostFunctionScaled(const Eigen::Vector3d &m2,
                                                            const Eigen::Vector3d &m1,
                                                            const Eigen::Vector3d &m0,
                                                            const Eigen::Matrix3d &R2,
                                                            const Eigen::Matrix3d &R1,
                                                            const Eigen::Matrix3d &R0) :
    _m2(m2),
    _m1(m1),
    _m0(m0),
    _R2(R2),
    _R1(R1),
    _R0(R0) {}

template<typename T>
bool IterativeRefinement::CostFunctionScaled::operator()(const T *vec0, const T *vec1, const T *scale0, const T *scale1,
                                                         T *residuals) const {
  T n0 = scaleTemplated(scale0[0]);
  T n1 = scaleTemplated(scale1[0]);

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
  A << T(1.0) - a * a, T(-2.0) * a, T(0),
      T(2) * a, T(1.0) - a * a, T(0),
      T(0), T(0), T(1);

  Eigen::Matrix<T, 3, 3> B;
  B << T(1) - b * b, T(0), T(2) * b,
      T(0), T(1), T(0),
      T(2.0) * b, T(0), T(1.0) - b * b;

  return ((A * B) / ((1.0 + a * a) * (1.0 + b * b))) * vec.template cast<T>();

}

template<typename T>
T IterativeRefinement::scaleTemplated(T t) {
  return LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1.0 + ceres::exp(-1.0 * t)));
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

double IterativeRefinement::reverseScale(const double length) {
  double t = 1;
  if (length <=
      LOW_VALUE) //Catch the Cases in which the SCaling is to low or high, cause it is Mathematical impossible to calc t
  {
    t = reverseScale(LOW_VALUE + std::numeric_limits<double>::epsilon());
    ROS_WARN_STREAM("Lower bound of scaling to high: " << length << std::endl);
  } else if (length >= HIGH_VALUE) {
    t = reverseScale(HIGH_VALUE - std::numeric_limits<double>::epsilon());
    ROS_WARN_STREAM("Upper bound of scaling to low: " << length << std::endl);
  } else {
    t = -1.0 * std::log((HIGH_VALUE - length) / (length - LOW_VALUE));
  }
  return t;
}