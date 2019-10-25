#include "IterativeRefinement.hpp"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>
#include <exception>
#include <typeinfo>
#include <string>
#include <sstream>
#include <iostream>

void IterativeRefinement::refine(std::vector<RefinementFrame> &refinementData,
                                 unsigned int numberToRefine,
                                 unsigned int numberToNote,
                                 int maxNumthreads,
                                 int maxNumIterations,
                                 double functionTolerance,
                                 double gradientTolerance,
                                 double parameterTolerance,
                                 bool useLossFunction,
                                 double lowestLength,
                                 double highestLength) {
  assert(refinementData.size() == numberToNote);
  assert(numberToNote > numberToRefine);
  assert(numberToRefine > 0);
  //Convert into EIGEN-Space:
  std::vector<RefinementFrameEIG> frames(refinementData.size());
  auto cvFrame = refinementData.begin();
  auto eigFrame = frames.begin();
  for (; cvFrame != refinementData.end(); cvFrame++, eigFrame++) {
    cvt_cv_eigen(cvFrame->m, eigFrame->m);
    cvt_cv_eigen(cvFrame->R, eigFrame->R);
    cvt_cv_eigen(cvFrame->vec, eigFrame->vec);
    eigFrame->scale = cvFrame->scale;
  }


  //Create Params
  std::vector<double[1]> scales(numberToRefine);
  std::vector<double[3]> vectors(numberToRefine);



  /*double scale0[1] = {reverseScale(refinementData.scale0, highestLength, lowestLength)};
  double scale1[1] = {reverseScale(refinementData.scale1, highestLength, lowestLength)};

  double vec0[3] = {dataEIG.vec0(0), dataEIG.vec0(1), dataEIG.vec0(2)};
  double vec1[3] = {dataEIG.vec1(0), dataEIG.vec1(1), dataEIG.vec1(2)};

  ROS_INFO_STREAM("Before: ");
  ROS_INFO_STREAM("n0 * u0: " << refinementData.scale0 << " * [" << vec0[0] << "," << vec0[1] << "," << vec0[2] << "]");
  ROS_INFO_STREAM("n1 * u1: " << refinementData.scale1 << " * [" << vec1[0] << "," << vec1[1] << "," << vec1[2] << "]");*/

  //Set Option:
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

  //Create Params:
  ROS_INFO_STREAM("BEFORE:");
  for (unsigned int frameID = 0; frameID < numberToRefine; frameID++) {
    vectors[frameID][0] = frames[frameID].vec[0];
    vectors[frameID][1] = frames[frameID].vec[1];
    vectors[frameID][2] = frames[frameID].vec[2];
    ROS_INFO_STREAM("[" << frameID << "] - " << frames[frameID].scale << " * "  << refinementData[frameID].vec);
    scales[frameID][0] = reverseScale(frames[frameID].scale, highestLength, lowestLength);

    ceres_problem.AddParameterBlock(vectors[frameID], 3);
    ceres_problem.AddParameterBlock(scales[frameID], 1);

    ceres::LocalParameterization
        *local_parametrizationVec = new ceres::AutoDiffLocalParameterization<ParametrizedBaseLine, 3, 2>;
    ceres_problem.SetParameterization(vectors[frameID], local_parametrizationVec);
  }


  //Go through the Frames:
  for (unsigned int frame0 = 0; frame0 < numberToRefine; frame0++) {
    //Add the "normal" CostFunction
    ceres::LossFunction *lossFun = nullptr;
    if (useLossFunction) {
      lossFun = new ceres::CauchyLoss(0.5);
    }
    CostFunction::addResidualBlocks(frames[frame0 + 1].m,
                                    frames[frame0].m,
                                    frames[frame0 + 1].R,
                                    frames[frame0].R,
                                    lossFun,
                                    vectors[frame0],
                                    ceres_problem);

    //Iterate through each following Frame to create a correspondence to each Frame
    std::vector<double *> parameter_blocks;
    Eigen::Vector3d vector_offset(0, 0, 0);
    parameter_blocks.push_back(scales[frame0]);
    parameter_blocks.push_back(vectors[frame0]);
    int params_counter = 1;
    for (unsigned int frame1 = (frame0 + 1); frame1 < (numberToNote - 1);
         frame1++) {  //Skip Last Frame, cause its Vector an dscale to prev is unneccecccary
      if (frame1 < numberToRefine) { //Frame1 also needs to be Refined
        parameter_blocks.push_back(scales[frame1]);
        parameter_blocks.push_back(vectors[frame1]);
        params_counter++;
      } else { //Don't Refine Frame1
        vector_offset = vector_offset + frames[frame1].scale * frames[frame1].vec;
      }
      //Add this "scaled" CostFunction
      CostFunctionScaled::addResidualBlocks(frames[frame1 + 1].m,
                                            frames[frame0].m,
                                            frames[frame1 + 1].R,
                                            frames[frame0].R,
                                            lossFun,
                                            parameter_blocks,
                                            ceres_problem,
                                            vector_offset,
                                            lowestLength,
                                            highestLength,
                                            params_counter);

    }

  }

  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options, &ceres_problem, &ceres_summary
  );


  if (ceres_summary.termination_type == ceres::TerminationType::FAILURE) {
    throw std::runtime_error("CERES ERROR!");
  }

  ROS_INFO_STREAM("AFTER:");
  for (unsigned int i = 0; i < numberToRefine; i++) {
    auto scale = scaleTemplated<double>(scales[i][0], highestLength, lowestLength);
    cv::Vec3d vec = cvt_eigen_cv(Eigen::Vector3d(vectors[i][0], vectors[i][1], vectors[i][2]));
    refinementData[i].scale = scale;
    refinementData[i].vec = vec;
    ROS_INFO_STREAM("[" << i << "] - " << scale << " * "  << vec);
  }

  ROS_INFO_STREAM(ceres_summary.FullReport());

}

IterativeRefinement::CostFunctionScaled::CostFunctionScaled(const Eigen::Vector3d &m1,
                                                            const Eigen::Vector3d &m0,
                                                            const Eigen::Matrix3d &R1,
                                                            const Eigen::Matrix3d &R0,
                                                            const Eigen::Vector3d &vectorOffset,
                                                            double maxLength,
                                                            double minLength,
                                                            int params) :
    _m1(m1),
    _m0(m0),
    _R1(R1),
    _R0(R0),
    _vectorOffset(vectorOffset),
    _maxLength(maxLength),
    _minlength(minLength),
    _params(params) {}

template<typename T>
bool IterativeRefinement::CostFunctionScaled::operator()(T const *const *parameters,
                                                         T *residuals) const {
  //Calculate the "baseLine"
  Eigen::Matrix<T, 3, 1> u01 = _vectorOffset.template cast<T>();

  for (int i = 0; i < _params; i++) {
    T scale = scaleTemplated(parameters[i][0], _maxLength, _minlength);
    Eigen::Matrix<T, 3, 1> vector(parameters[i + 1][0], parameters[i + 1][1], parameters[i + 1][2]);
    u01 = u01 + scale * vector;
  }
  //Normalize the "baseLine";
  u01.normalize();

  T cost = ((_m1.template cast<T>()).dot(
      (_R1).transpose().template cast<T>() * u01.cross((_R0).template cast<T>() * (_m0).template cast<T>())));
  residuals[0] = cost;

  return true;
}

void IterativeRefinement::CostFunctionScaled::addResidualBlocks(const std::vector<Eigen::Vector3d> &m1,
                                                                const std::vector<Eigen::Vector3d> &m0,
                                                                const Eigen::Matrix3d &R1,
                                                                const Eigen::Matrix3d &R0,
                                                                ceres::LossFunction *lossFunction,
                                                                std::vector<double *> &parameter_blocks,
                                                                ceres::Problem &ceresProblem,
                                                                Eigen::Vector3d vector_offset,
                                                                double minLength,
                                                                double maxLength,
                                                                int params) {
  for (unsigned int i = 0; i < m1.size(); i++) {
    ceres::DynamicAutoDiffCostFunction<CostFunctionScaled, 4> //TODO: Number of Stride!!
        *scaled_costfunction = new ceres::DynamicAutoDiffCostFunction<CostFunctionScaled, 4>(
        new CostFunctionScaled(m1[i], m0[i], R1, R0, vector_offset, maxLength, minLength, params)
    );

    for (int j = 0; j < params; j++) {
      scaled_costfunction->AddParameterBlock(1);
      scaled_costfunction->AddParameterBlock(3);
    }
    scaled_costfunction->SetNumResiduals(1);
    ceresProblem.AddResidualBlock(scaled_costfunction,
                                  lossFunction,
                                  parameter_blocks); //TODO: Parametersblock vector copyed?
  }
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

void IterativeRefinement::CostFunction::addResidualBlocks(const std::vector<Eigen::Vector3d> &m1,
                                                          const std::vector<Eigen::Vector3d> &m0,
                                                          const Eigen::Matrix3d &R1,
                                                          const Eigen::Matrix3d &R0,
                                                          ceres::LossFunction *lossFunction,
                                                          double *vectorParam, ceres::Problem &ceresProblem) {
  for (unsigned int i = 0; i < m1.size(); i++) {
    ceres::CostFunction *costfunctionUnscaled = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
        new CostFunction(m1[i], m0[i], R1, R0)
    );
    ceresProblem.AddResidualBlock(costfunctionUnscaled, lossFunction, vectorParam);
  }
}

template<typename T>
Eigen::Matrix<T, 3, 1> IterativeRefinement::baseLineTemplated(const Eigen::Matrix<T, 3, 1> &vec, const T a, const T b) {
  Eigen::Matrix<T, 3, 3> A;
  A << (T(1.0) - a * a) / (T(1.0) + a * a), (T(-2.0) * a) / (T(1.0) + a * a), T(0),
      (T(1.0) * a) / (T(1.0) + a * a), (T(1.0) - a * a) / (T(1.0) + a * a), T(0),
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
  if (ceres::IsInfinite(exp)) { //In Case, that this term gets infinite. The whole function is instable for derivations i guess. (It results in "nan" in ceres. Therefore we have to cattch need to know, that 1/Inf ~= 0. So we have to return MIN_VALUE.
    result = T(MIN_LEN);
  } else {
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
    t = reverseScale(
        MIN_LEN + (length * std::numeric_limits<double>::epsilon()) + std::numeric_limits<double>::epsilon(),
        MAX_LEN,
        MIN_LEN);
    ROS_WARN_STREAM("Lower bound of scaling to high: " << length);
  } else if (length >= MAX_LEN) {
    t = reverseScale(
        MAX_LEN - (length * std::numeric_limits<double>::epsilon() + std::numeric_limits<double>::epsilon()),
        MAX_LEN,
        MIN_LEN);
    ROS_WARN_STREAM("Upper bound of scaling to low: " << length);
  } else {
    t = -1.0 * std::log((MAX_LEN - length) / (length - MIN_LEN));
  }
  return t;
}