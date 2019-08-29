#include "IterativeRefinement.hpp"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>
#include <ceres/ceres.h>


// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(SlidingWindow& slidingWindow) : _slidingWindow(slidingWindow)
{
}
IterativeRefinement::~IterativeRefinement()
{
}



void IterativeRefinement::refine(unsigned int n)
{

  _slidingWindow.exportMatlabData();
  ROS_INFO_STREAM("BEFORE: " << std::endl
                             << "st0: " << _slidingWindow.getPosition(0) << std::endl
                             << "st1: " << _slidingWindow.getPosition(1) << std::endl
                             << "st2: " << _slidingWindow.getPosition(2) << std::endl);

  assert(n == 3);  // TODO: currently only WindowSize 3 available
  RefinementDataCV data;

  cv::Vec3d& st0 = _slidingWindow.getPosition(0);
  cv::Vec3d& st1 = _slidingWindow.getPosition(1);
  cv::Vec3d& st2 = _slidingWindow.getPosition(2);

  double n0 = cv::norm(st0 - st1);
  double n1 = cv::norm(st1 - st2);
  cv::Vec3d u0 = (st0 - st1) / n0;
  cv::Vec3d u1 = (st1 - st2) / n1;

  ROS_INFO_STREAM("Before: " << std::endl);
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * " << u0 << std::endl);
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * " << u1 << std::endl);

  double params0[3] = {0,0,1};
  double params1[3] = {0,0,1};

  data.vec0 = u0;
  data.vec1 = u1; 

  data.R0 = _slidingWindow.getRotation(0);
  data.R1 = _slidingWindow.getRotation(1);
  data.R2 = _slidingWindow.getRotation(2);

  std::vector<std::vector<cv::Vec3d>*> vectors{ &(data.m0), &(data.m1), &(data.m2) };

  _slidingWindow.getCorrespondingFeatures(n - 1, 0, vectors);

  RefinementDataEIG dataEIG;
  cvt_cv_eigen(data.m0, dataEIG.m0);
  cvt_cv_eigen(data.m1, dataEIG.m1);
  cvt_cv_eigen(data.m2, dataEIG.m2);
  cvt_cv_eigen(data.R0, dataEIG.R0);
  cvt_cv_eigen(data.R1, dataEIG.R1);
  cvt_cv_eigen(data.R2, dataEIG.R2);
  cvt_cv_eigen(data.vec0, dataEIG.vec0);
  cvt_cv_eigen(data.vec1, dataEIG.vec1);



  ceres::Problem ceres_problem;
  ceres::Solver::Options ceres_solver_options;
  ceres_solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  ceres_solver_options.linear_solver_type = ceres::DENSE_QR;
  ceres_solver_options.max_num_iterations = 100;
  ceres_solver_options.num_threads = 8;
  //ceres_solver_options.check_gradients = true; ///DEBUG!

  for(unsigned int i = 0; i < data.m0.size(); i++){
    ceres::CostFunction* cost_functiom20 = new ceres::AutoDiffCostFunction<CostFunctionScaled, 1, 3, 3>(
      new CostFunctionScaled(dataEIG.m2[i], dataEIG.m1[i], dataEIG.m0[i], dataEIG.R2, dataEIG.R1, dataEIG.R0,dataEIG.vec0, dataEIG.vec1)
    );
    ceres::CostFunction* cost_functiom21 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
      new CostFunction(dataEIG.m2[i], dataEIG.m1[i], dataEIG.R2, dataEIG.R1,dataEIG.vec1)
    );
    ceres::CostFunction* cost_functiom10 = new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
      new CostFunction(dataEIG.m1[i], dataEIG.m0[i], dataEIG.R1, dataEIG.R0,dataEIG.vec0)
    );

    ceres_problem.AddResidualBlock(cost_functiom20, NULL, params0, params1);
    ceres_problem.AddResidualBlock(cost_functiom21, NULL, params1);
    ceres_problem.AddResidualBlock(cost_functiom10, NULL, params0);
  }
  

  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_solver_options, &ceres_problem, &ceres_summary);

  ROS_INFO_STREAM(ceres_summary.FullReport() << std::endl);
 

  n0 = scaleTemplated<double>(params0[2]);  // T0
  n1 = scaleTemplated<double>(params1[2]);  // T1

  u0 = cvt_eigen_cv(baseLineTemplated<double>(dataEIG.vec0, params0[0], params0[1]));
  u1 = cvt_eigen_cv(baseLineTemplated<double>(dataEIG.vec1, params1[0], params1[1]));

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


IterativeRefinement::CostFunctionScaled::CostFunctionScaled(const Eigen::Vector3d & m2,
                                                            const Eigen::Vector3d & m1,
                                                            const Eigen::Vector3d & m0,
                                                            const Eigen::Matrix3d & R2,
                                                            const Eigen::Matrix3d & R1,
                                                            const Eigen::Matrix3d & R0,
                                                            const Eigen::Vector3d & vec0,
                                                            const Eigen::Vector3d & vec1):
     _m2(m2),
     _m1(m1),
     _m0(m0),
     _R2(R2),
     _R1(R1),
     _R0(R0),
     _vec0(vec0),
     _vec1(vec1)
     {}


template <typename T>
bool IterativeRefinement::CostFunctionScaled::operator()(const T* parameters0, const T* parameters1, T* residuals) const{
  T a0 = parameters0[0];
  T b0 = parameters0[1];
  T t0 = parameters0[2];
  T a1 = parameters1[0];
  T b1 = parameters1[1];
  T t1 = parameters1[2];

  T n0 = scaleTemplated(t0);
  T n1 = scaleTemplated(t1);

  Eigen::Matrix<T,3,1>  u0 = baseLineTemplated(_vec0, a0, b0);
  Eigen::Matrix<T,3,1>  u1 = baseLineTemplated(_vec1, a1, b1);

  Eigen::Matrix<T,3,1>  u01 = n1 * u1 + n0 * u0;
  u01.normalize();

  T cost = ((_m2).transpose().template cast<T>() * (_R2).transpose().template cast<T>() * u01.cross((_R0).template cast<T>() * (_m0).template cast<T>())).value();  
  residuals[0] = cost;
  return true;
}



IterativeRefinement::CostFunction::CostFunction(const Eigen::Vector3d & m1,
                                                const Eigen::Vector3d & m0,
                                                const Eigen::Matrix3d & R1,
                                                const Eigen::Matrix3d & R0,
                                                const Eigen::Vector3d & vec):
     _m1(m1),
     _m0(m0),
     _R1(R1),
     _R0(R0),
     _vec(vec)
     {}

template <typename T>
bool IterativeRefinement::CostFunction::operator()(const T* parameters, T* residuals) const{
   T a = parameters[0];
   T b = parameters[1];

  Eigen::Matrix<T,3,1>  u = baseLineTemplated(_vec, a, b);

  T cost = (_m1.transpose().template cast<T>() * _R1.transpose().template cast<T>() * u.cross(_R0.template cast<T>() * _m0.template cast<T>())).value();
  residuals[0] = cost;
  return true;
}

template <typename T>
Eigen::Matrix<T,3,1> IterativeRefinement::baseLineTemplated(const Eigen::Vector3d & vec, const T a, const T b){
  Eigen::Matrix<T,3,3> A;
  A <<  T(1.0) - a*a , T(-2.0) * a, T(0),
  T(2)*a       , T(1.0) - a*a,  T(0),
    T(0)         , T(0)        ,  T(1);

   Eigen::Matrix<T,3,3> B;
   B << T(1) - b*b   ,T(0)       ,  T(2)*b,
   T(0)         ,T(1)       ,   T(0),
   T(2.0)*b    ,T(0)       , T(1.0) - b*b;

  return ((A * B) / ((1.0 + a * a) * (1.0 + b* b))) * vec.template cast<T>();

}

template <typename T>
T IterativeRefinement::scaleTemplated(T t){
   return LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1.0 + ceres::exp(-1.0 * t)));
}




void IterativeRefinement::cvt_cv_eigen(const std::vector<cv::Vec3d> & vecaCV, std::vector<Eigen::Vector3d> & vecaEIGEN){
  vecaEIGEN.resize(vecaCV.size());
  auto cvIT = vecaCV.begin();
  auto eiIT = vecaEIGEN.begin();
  for(;cvIT != vecaCV.end(); cvIT++, eiIT++){
    *eiIT << (*cvIT)(0),(*cvIT)(1),(*cvIT)(2); 
  }
}
void IterativeRefinement::cvt_cv_eigen(const cv::Matx33d & matCV, Eigen::Matrix3d & matEIG){
  matEIG << matCV(0,0), matCV(0,1), matCV(0,2),
            matCV(1,0), matCV(1,1), matCV(1,2),
            matCV(2,0), matCV(2,1), matCV(2,2);
}

void IterativeRefinement::cvt_cv_eigen(const cv::Vec3d & vecCV, Eigen::Vector3d & vecEIG){
  vecEIG << vecCV(0), vecCV(1), vecCV(2);
}

cv::Vec3d IterativeRefinement::cvt_eigen_cv(const Eigen::Vector3d & vecEIG){
  return cv::Vec3d(vecEIG(0), vecEIG(1), vecEIG(2));
}