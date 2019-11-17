#include "IterativeRefinement.hpp"
#include "../operations/FeatureOperations.h"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>
#include <exception>
#include <typeinfo>
#include <string>
#include <sstream>
#include <iostream>
#include <list>

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
                                 double highestLength,
                                 const Frame &nowFrame
) {
  assert(refinementData.size() == numberToNote);
  assert(numberToNote > numberToRefine);
  assert(numberToRefine > 0);
  ROS_INFO_STREAM(
      "REFINEMENT: " << std::endl << "\trefine: " << numberToRefine << std::endl << "\tnote: " << numberToNote);
  //Convert into EIGEN-Space:
  std::vector<RefinementFrameEIG> frames(refinementData.size());
  auto cvFrame = refinementData.begin();
  auto eigFrame = frames.begin();
  for (; cvFrame != refinementData.end(); cvFrame++, eigFrame++) {
    cvt_cv_eigen(cvFrame->R, eigFrame->R);
    cvt_cv_eigen(cvFrame->vec, eigFrame->vec);
    eigFrame->scale = cvFrame->scale;
  }


  //Create Params
  std::vector<std::array<double, 1>> scales(numberToRefine);
  std::vector<std::array<double, 3>> vectors(numberToRefine);



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
  //ceres_solver_options.use_explicit_schur_complement = true;
  ceres_solver_options.max_num_iterations = maxNumIterations;
  ceres_solver_options.num_threads = maxNumthreads;
  //ceres_solver_options.use_inner_iterations = true;
  ceres_solver_options.use_nonmonotonic_steps = true;
  ceres_solver_options.preconditioner_type = ceres::JACOBI;
  ceres_solver_options.function_tolerance = functionTolerance;
  ceres_solver_options.gradient_tolerance = gradientTolerance;
  ceres_solver_options.parameter_tolerance = parameterTolerance;
  //ceres_solver_options.check_gradients = true; ///DEBUG!
  //ceres_solver_options.minimizer_progress_to_stdout = true; ///DEBUG!

  //Create Local Parametrization
  ceres::LocalParameterization
      *local_parametrization_vec = new ceres::AutoDiffLocalParameterization<ParametrizedBaseLine, 3, 2>;

  //Create loss_function if required
  ceres::LossFunction *loss_function = nullptr;
  if (useLossFunction) {
    loss_function = new ceres::CauchyLoss(0.5);
  }



  //Create Params:
  ROS_INFO_STREAM("BEFORE:");
  // Create Vectors for frame 0 until numbertoNote - 1, because the Last frames contains only the baseline to a not to Note Frame
  for (unsigned int frameID = 0; frameID < (numberToNote - 1); frameID++) {
    //If this Frame should not be Refined, than "block" it.
    if (frameID < numberToRefine) {
      ROS_INFO_STREAM("[" << frameID << "] - " << frames[frameID].scale << " * " << refinementData[frameID].vec);
      vectors[frameID][0] = frames[frameID].vec[0];
      vectors[frameID][1] = frames[frameID].vec[1];
      vectors[frameID][2] = frames[frameID].vec[2];
      scales[frameID][0] = reverseScale(frames[frameID].scale, highestLength, lowestLength);
      ceres_problem.AddParameterBlock(vectors[frameID].data(), 3, local_parametrization_vec);
      ceres_problem.AddParameterBlock(scales[frameID].data(), 1);
    }
  }

  std::list<std::vector<Eigen::Vector3d>> features; //To Store all Vectors while refinement (to avoid Copying)
  std::list<Eigen::Vector3d> vector_offsets;
  std::vector<std::vector<double *>> parameter_blocks;


  //Go through the Frames:
  for (unsigned int frame0 = 0; frame0 < numberToRefine; frame0++) {
#ifdef DEBUGIMAGES
    cv::Scalar color(255, 255, 255);
    switch (frame0) {
      case 0:color = cv::Scalar(0, 0, 255);
        break;
      case 1:color = cv::Scalar(0, 255, 0);
        break;
      case 2:color = cv::Scalar(255, 0, 0);
        break;
      case 3:color = cv::Scalar(0, 255, 255);
        break;

    }
#endif

    //Iterate through each following Frame to create a correspondence to each Frame
    parameter_blocks.push_back({});
    for (unsigned int frame1 = (frame0); frame1 < (numberToNote - 1);
         frame1++) {  //Skip Last Frame, cause its vector and scale to prev is unneccecccary. The Last Vector is implicit used, when receiving the Features of it.

      vector_offsets.push_back(Eigen::Vector3d(0, 0, 0));
      if (frame1 < numberToRefine) {
        //Add Paramert Block. It doesn't matters, wether it ist a "Note" or a "Refine" Frame. Cause we block them above.
        parameter_blocks.back().push_back(scales[frame1].data());
        parameter_blocks.back().push_back(vectors[frame1].data());
      } else {
        vector_offsets.back() += (frames[frame1].scale * frames[frame1].vec);
      }
      {//Get features
        std::vector<cv::Vec3d> feature1CV, feature0CV;
        features.push_back({});
        std::vector<Eigen::Vector3d> &feature1EIG = features.back();
        features.push_back({});
        std::vector<Eigen::Vector3d> &feature0EIG = features.back();

        nowFrame.getPreviousFrame(frame0).getCorrespondingFeatures((frame1 - frame0) + 1, feature1CV, feature0CV);
        FeatureOperations::normFeatures(feature0CV);
        FeatureOperations::normFeatures(feature1CV);

#ifdef DEBUGIMAGES
        VisualisationUtils::drawCorrespondences({&feature0CV, &feature1CV},
                                                nowFrame.getCameraModel(),
                                                _debugImage,
                                                color,
                                                color);
#endif

        cvt_cv_eigen(feature0CV, feature0EIG);
        cvt_cv_eigen(feature1CV, feature1EIG);


        //Add this "scaled" CostFunction
        addResidualBlocks(feature1EIG,
                          feature0EIG,
                          frames[frame1 + 1].R,
                          frames[frame0].R,
                          parameter_blocks.back(),
                          vector_offsets.back(),
                          loss_function,
                          ceres_problem,
                          highestLength,
                          lowestLength);
      }

    }

  }

 /* //The "new" CostFunction:
  if(numberToNote >= 3 && numberToRefine >= 2) {
    for (unsigned int frame0 = 0; frame0 <= (numberToRefine - 2); frame0++) {
      auto frame1 = frame0 + 1;
      auto frame2 = frame0 + 2;
      std::vector<cv::Vec3d> feature2CV, feature1CV, feature0CV;
      features.push_back({});
      std::vector<Eigen::Vector3d> &feature2EIG = features.back();
      features.push_back({});
      std::vector<Eigen::Vector3d> &feature1EIG = features.back();
      features.push_back({});
      std::vector<Eigen::Vector3d> &feature0EIG = features.back();
      Frame::getCorrespondingFeatures<cv::Vec3d>(nowFrame.getPreviousFrame(frame2),
                                                 nowFrame.getPreviousFrame(frame0),
                                                 {&feature0CV, &feature1CV, &feature2CV});
      FeatureOperations::normFeatures(feature0CV);
      FeatureOperations::normFeatures(feature1CV);
      FeatureOperations::normFeatures(feature2CV);
      cvt_cv_eigen(feature0CV, feature0EIG);
      cvt_cv_eigen(feature1CV, feature1EIG);
      cvt_cv_eigen(feature2CV, feature2EIG);

      auto m0 = feature0EIG.begin();
      auto m1 = feature1EIG.begin();
      auto m2 = feature2EIG.begin();
      for (; m0 != feature0EIG.end(); m0++, m1++, m2++) {
        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ScaleCostFunction, 1, 1, 3, 1, 3>(
            new ScaleCostFunction(*m2,
                                  *m1,
                                  *m0,
                                  frames[frame2].R,
                                  frames[frame1].R,
                                  frames[frame0].R,
                                  highestLength,
                                  lowestLength)
        );
        ceres_problem.AddResidualBlock(cost_function,
                                       nullptr,
                                       scales[frame0].data(),
                                       vectors[frame0].data(),
                                       scales[frame1].data(),
                                       vectors[frame1].data());
      }
    }
    }*/

  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options, &ceres_problem, &ceres_summary
  );

  if (ceres_summary.termination_type == ceres::TerminationType::FAILURE) {
    ROS_ERROR_STREAM("CERES ERROR!" << ceres_summary.FullReport());
  }

  ROS_INFO_STREAM("AFTER:");
  for (unsigned int i = 0; i < numberToRefine; i++) {
    double scale = scaleTemplated<double>(scales[i][0], highestLength, lowestLength);
    cv::Vec3d vec = cvt_eigen_cv(Eigen::Vector3d(vectors[i][0], vectors[i][1], vectors[i][2]));
    refinementData[i].scale = scale;
    refinementData[i].vec = vec;
    ROS_INFO_STREAM("[" << i << "] - " << scale << " * " << vec);
  }

  ROS_INFO_STREAM(ceres_summary.FullReport());

}

template<typename T>
T IterativeRefinement::CostFunctionBase::cost(const Eigen::Matrix<T, 3, 1> &baseLine) const {
  Eigen::Matrix<T, 3, 1> u01 = baseLine + _vectOffset.template cast<T>();
  u01.normalize();

  T cost = (_m1.template cast<T>().dot(
      (_R1.template cast<T>()).transpose() * (u01.cross(((_R0.template cast<T>()) * (_m0.template cast<T>()))))));

  return cost;
}

IterativeRefinement::CostFunctionBase::CostFunctionBase(const Eigen::Vector3d &m1,
                                                        const Eigen::Vector3d &m0,
                                                        const Eigen::Matrix3d &R1,
                                                        const Eigen::Matrix3d &R0,
                                                        double maxLength,
                                                        double minLength,
                                                        const Eigen::Vector3d &vectOffset) :
    _m1(m1),
    _m0(m0),
    _R1(R1),
    _R0(R0),
    _vectOffset(vectOffset),
    _maxLength(maxLength),
    _minlength(minLength) {

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
  T exp = ceres::exp(-1.0 * t);
  if (ceres::IsInfinite(exp)) { //In Case, that this term gets infinite. The whole function is instable for derivations i guess. (It results in "nan" in ceres. Therefore we have to catch. We know, that 1/Inf ~= 0. So we have to return MIN_VALUE.
    result = T(MIN_LEN);
    //ROS_ERROR_STREAM("INfinity Case");
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
void IterativeRefinement::addResidualBlocks(const std::vector<Eigen::Vector3d> &features1,
                                            const std::vector<Eigen::Vector3d> &features0,
                                            const Eigen::Matrix3d &R1,
                                            const Eigen::Matrix3d &R0,
                                            std::vector<double *> parameter_blocks,
                                            const Eigen::Vector3d &vect_offset,
                                            ceres::LossFunction *loss_fun,
                                            ceres::Problem &ceres_problem,
                                            double highest_len,
                                            double lowest_len) {
  assert(features0.size() == features1.size());
  auto f1 = features1.begin();
  auto f0 = features0.begin();
  for (; f1 != features1.end(); f1++, f0++) {
    ceres::CostFunction *cost_function;
    switch (parameter_blocks.size()) {
      case 2:
        cost_function = new ceres::AutoDiffCostFunction<CostFunction1, 1, 3>(
            new CostFunction1(*f1, *f0, R1, R0, highest_len, lowest_len, vect_offset)

        );
        break;
      case 4:
        cost_function = new ceres::AutoDiffCostFunction<CostFunction2, 1, 1, 3, 1, 3>(
            new CostFunction2(*f1, *f0, R1, R0, highest_len, lowest_len, vect_offset)
        );
        break;
      case 6:
        cost_function = new ceres::AutoDiffCostFunction<CostFunction3, 1, 1, 3, 1, 3, 1, 3>(
            new CostFunction3(*f1, *f0, R1, R0, highest_len, lowest_len, vect_offset)
        );
        break;
      case 8:
        cost_function = new ceres::AutoDiffCostFunction<CostFunction4, 1, 1, 3, 1, 3, 1, 3, 1, 3>(
            new CostFunction4(*f1, *f0, R1, R0, highest_len, lowest_len, vect_offset)
        );
        break;
      case 10:
        cost_function = new ceres::AutoDiffCostFunction<CostFunction5, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3>(
            new CostFunction5(*f1, *f0, R1, R0, highest_len, lowest_len, vect_offset)
        );
        break;
      default:assert(false); //FAIL!
    };
    ceres_problem.AddResidualBlock(cost_function, loss_fun, parameter_blocks);
  }

}

IterativeRefinement::CostFunction1::CostFunction1(
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    double max_length,
    double min_length,
    const Eigen::Vector3d &vect_offset) : CostFunctionBase(m1, m0, r1, r0, max_length, min_length, vect_offset) {}

template<typename T>
bool IterativeRefinement::CostFunction1::operator()(const T *vec0, T *residuals) const {
  residuals[0] = CostFunctionBase::cost(Eigen::Matrix<T, 3, 1>(
      Eigen::Matrix<T, 3, 1>(vec0[0], vec0[1], vec0[2])).eval());
  return true;
}

IterativeRefinement::CostFunction2::CostFunction2(
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    double max_length,
    double min_length,
    const Eigen::Vector3d &vect_offset) : CostFunctionBase(m1, m0, r1, r0, max_length, min_length, vect_offset) {}

template<typename T>
bool IterativeRefinement::CostFunction2::operator()(const T *scale0,
                                                    const T *vec0,
                                                    const T *scale1,
                                                    const T *vec1,
                                                    T *residuals) const {
  residuals[0] = CostFunctionBase::cost(Eigen::Matrix<T, 3, 1>(
      scaleTemplated(scale0[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec0[0], vec0[1], vec0[2]) +
          scaleTemplated(scale1[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec1[0], vec1[1], vec1[2])).eval()
  );
  return true;
}

IterativeRefinement::CostFunction3::CostFunction3(
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    double max_length,
    double min_length,
    const Eigen::Vector3d &vect_offset) : CostFunctionBase(m1, m0, r1, r0, max_length, min_length, vect_offset) {}

template<typename T>
bool IterativeRefinement::CostFunction3::operator()(const T *scale0,
                                                    const T *vec0,
                                                    const T *scale1,
                                                    const T *vec1,
                                                    const T *scale2,
                                                    const T *vec2,
                                                    T *residuals) const {
  residuals[0] = CostFunctionBase::cost(Eigen::Matrix<T, 3, 1>(
      scaleTemplated(scale0[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec0[0], vec0[1], vec0[2]) +
          scaleTemplated(scale1[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec1[0], vec1[1], vec1[2]) +
          scaleTemplated(scale2[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec2[0], vec2[1], vec2[2])).eval()
  );
  return true;
}

IterativeRefinement::CostFunction4::CostFunction4(
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    double max_length,
    double min_length,
    const Eigen::Vector3d &vect_offset) : CostFunctionBase(m1, m0, r1, r0, max_length, min_length, vect_offset) {}

template<typename T>
bool IterativeRefinement::CostFunction4::operator()(const T *scale0,
                                                    const T *vec0,
                                                    const T *scale1,
                                                    const T *vec1,
                                                    const T *scale2,
                                                    const T *vec2,
                                                    const T *scale3,
                                                    const T *vec3,
                                                    T *residuals) const {
  Eigen::Matrix<T, 3, 1> vec(
      scaleTemplated(scale0[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec0[0], vec0[1], vec0[2]) +
          scaleTemplated(scale1[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec1[0], vec1[1], vec1[2]) +
          scaleTemplated(scale2[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec2[0], vec2[1], vec2[2]) +
          scaleTemplated(scale3[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec3[0], vec3[1], vec3[2]));
  residuals[0] = CostFunctionBase::cost(vec);
  return true;
}

IterativeRefinement::CostFunction5::CostFunction5(
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    double max_length,
    double min_length,
    const Eigen::Vector3d &vect_offset) : CostFunctionBase(m1, m0, r1, r0, max_length, min_length, vect_offset) {}

template<typename T>
bool IterativeRefinement::CostFunction5::operator()(const T *scale0,
                                                    const T *vec0,
                                                    const T *scale1,
                                                    const T *vec1,
                                                    const T *scale2,
                                                    const T *vec2,
                                                    const T *scale3,
                                                    const T *vec3,
                                                    const T *scale4,
                                                    const T *vec4,
                                                    T *residuals) const {
  Eigen::Matrix<T, 3, 1>
      vec(scaleTemplated(scale0[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec0[0], vec0[1], vec0[2]) +
      scaleTemplated(scale1[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec1[0], vec1[1], vec1[2]) +
      scaleTemplated(scale2[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec2[0], vec2[1], vec2[2]) +
      scaleTemplated(scale3[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec3[0], vec3[1], vec3[2]) +
      scaleTemplated(scale4[0], _maxLength, _minlength) * Eigen::Matrix<T, 3, 1>(vec4[0], vec4[1], vec4[2]));
  residuals[0] = cost(vec);

  return true;
}
IterativeRefinement::ScaleCostFunction::ScaleCostFunction(
    const Eigen::Vector3d &m2,
    const Eigen::Vector3d &m1,
    const Eigen::Vector3d &m0,
    const Eigen::Matrix3d &r2,
    const Eigen::Matrix3d &r1,
    const Eigen::Matrix3d &r0,
    const double max_length,
    const double min_length)
    : _m2(m2), _m1(m1), _m0(m0), _R2(r2), _R1(r1), _R0(r0), _maxLength(max_length), _minlength(min_length) {}

template<typename T>
bool IterativeRefinement::ScaleCostFunction::operator()(const T *scale0,
                                                        const T *vec0,
                                                        const T *scale1,
                                                        const T *vec1,
                                                        T *residuals) const {
  T n0 = scaleTemplated(scale0[0], _maxLength, _minlength);
  T n1 = scaleTemplated(scale1[1], _maxLength, _minlength);

  Eigen::Matrix<T, 3, 1> u0(vec0[0], vec0[1], vec0[2]);
  Eigen::Matrix<T, 3, 1> u1(vec1[0], vec1[1], vec1[2]);

  T cost = n1 * ceres::sqrt(T(1.0) - ceres::pow((_R2 * _m2).template cast<T>().dot(u1), 2))
      * ceres::sqrt(1 - ceres::pow((_R1 * _m1).dot(_R0 * _m0), 2))
      - n0 * ceres::sqrt(1 - ceres::pow((_R2 * _m2).dot(_R1 * _m1), 2))
          * ceres::sqrt(T(1.0) - ceres::pow(u0.dot((_R0 * _m0).template cast<T>()), 2));

  residuals[0] = cost;
  return true;
}
