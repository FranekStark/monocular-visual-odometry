#include "IterativeRefinement.hpp"
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

#include <limits>

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(SlidingWindow& slidingWindow) : _slidingWindow(slidingWindow)
{
}
IterativeRefinement::~IterativeRefinement()
{
}

double IterativeRefinement::CostFunction::func10(const Input & input, double a, double b, double x, double y, double z){
  return func(input.m1, input.R1,baseLine(a,b,x,y,z), input.R0, input.m0);
}
double IterativeRefinement::CostFunction::func21(const Input & input, double a1, double b1, double x1, double y1, double z1){
  return func(input.m2, input.R2, baseLine(a1,b1,x1,y1,z1), input.R1, input.m1);
}
double IterativeRefinement::CostFunction::func20(const Input & input, double a, double b, double x, double y, double z, double t, double a1, double b1, double x1, double y1, double z1, double t1){
  auto u = CostFunction::scale(t)*CostFunction::baseLine(a,b,x,y,z)+CostFunction::scale(t1)*CostFunction::baseLine(a1,b1,x1,y1,z1);
  return CostFunction::func(input.m2,input.R2,
    u/cv::norm(u)
    ,input.R0, input.m0);
}

double IterativeRefinement::derivationParam(double x){
  if(x == 0){
    return std::sqrt(DBL_EPSILON);
  }
  return x * std::sqrt(DBL_EPSILON);
}


void IterativeRefinement::CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const RefinementData& data,
                                                    const cv::Mat& params)
{
  assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());
  for (unsigned int i = 0; i < data.m0.size(); i++)
  {  // TODO: Faster Access
    double a = params.at<double>(0,0);
    double b = params.at<double>(1,0);
    double t = params.at<double>(2,0);
    double a1 = params.at<double>(3,0);
    double b1 = params.at<double>(4,0);
    double t1 = params.at<double>(5,0);
    double x = params.at<double>(0,1);
    double y = params.at<double>(1,1);
    double z = params.at<double>(2,1);
    double x1 = params.at<double>(3,1);
    double y1 = params.at<double>(4,1);
    double z1 = params.at<double>(5,1);

    const cv::Vec3d & m0 = data.m0[i];
    const cv::Vec3d & m1 = data.m1[i];
    const cv::Vec3d & m2 = data.m2[i];
    const cv::Matx33d & R0 = data.R0;
    const cv::Matx33d & R1 = data.R1;
    const cv::Matx33d & R2 = data.R2;

    double cost = CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b,t);

  
    double deriveA =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a + derivationParam(a),b,t) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a - derivationParam(a),b,t)) /
                      (2.0 * derivationParam(a));
    double deriveB =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b + derivationParam(b),t) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b - derivationParam(b),t)) /
                      (2.0 * derivationParam(b));
    double deriveT =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b ,t + derivationParam(t)) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b ,t - derivationParam(t))) /
                      (2.0 * derivationParam(t));

    double deriveA1 =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1 + derivationParam(a1),b1,t1,a,b,t) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1- derivationParam(a1),b1,t1,a ,b,t)) /
                      (2.0 * derivationParam(a1));
    double deriveB1 =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1+ derivationParam(b1),t1,a,b ,t) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1- derivationParam(b1),t1,a,b ,t)) /
                      (2.0 * derivationParam(b1));
    double deriveT1 =  (CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1+ derivationParam(t1),a,b ,t ) -
                      CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1- derivationParam(t1),a,b ,t )) /
                      (2.0 * derivationParam(t1));


    J.at<double>(i,0) = deriveA; J.at<double>(i,1) = deriveB; J.at<double>(i,2) = deriveT; J.at<double>(i,3) = deriveA1; J.at<double>(i,4) = deriveB1; J.at<double>(i,5) = deriveT1; 
    F.at<double>(i,0) = cost;
   
  }
}

cv::Mat IterativeRefinement::CreateFunction(const RefinementData& data, const cv::Mat& params)
{
  assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());
  cv::Mat f(data.m0.size(), 1, CV_64F);
  for (unsigned int i = 0; i < data.m1.size(); i++)
  {  // TODO: Faster Access
     double a = params.at<double>(0,0);
    double b = params.at<double>(1,0);
    double t = params.at<double>(2,0);
    double a1 = params.at<double>(3,0);
    double b1 = params.at<double>(4,0);
    double t1 = params.at<double>(5,0);
    double x = params.at<double>(0,1);
    double y = params.at<double>(1,1);
    double z = params.at<double>(2,1);
    double x1 = params.at<double>(3,1);
    double y1 = params.at<double>(4,1);
    double z1 = params.at<double>(5,1);

    const cv::Vec3d & m0 = data.m0[i];
    const cv::Vec3d & m1 = data.m1[i];
    const cv::Vec3d & m2 = data.m2[i];
    const cv::Matx33d & R0 = data.R0;
    const cv::Matx33d & R1 = data.R1;
    const cv::Matx33d & R2 = data.R2;

    double cost = CostFunction::funcWhole(m2,R2,x1,y1,z1,R1,m1,x,y,z,R0,m0,a1,b1,t1,a,b,t);

     f.at<double>(i,0) = cost;
  }
  return f;
}

/*Isn't Gauss-Newotn! -> It's now Levemberg Marquardt... */
// TODO: From https://www.mrpt.org/Levenberg-Marquardt_algorithm
void IterativeRefinement::GaussNewton(const RefinementData& data, cv::Mat& params)
{
 assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());
  unsigned int n = data.m0.size();
  double lambda = 10.0E-10; //Damping
  double thetha = 0.1; //tolerance

  cv::Mat J(n, 6, CV_64F);
  cv::Mat F(n,1,CV_64F);
  cv::Mat delta(6,1,CV_64F);
  do{
    CreateJacobianAndFunction(J, F, data, params);
    cv::Mat H = J.t() * J;
    cv::Mat gradient = J.t() * F;
    cv::solve(H + lambda * cv::Mat::eye(H.size(), CV_64F), gradient, delta, cv::DECOMP_CHOLESKY); //Was QR

      cv::Mat newParams = params.col(0) + delta;


      auto newF = CreateFunction(data, newParams);
      ROS_INFO_STREAM("F:" << std::endl << F <<std::endl << "newF:" << std::endl << newF << std::endl);
      if(cv::norm(newF) < cv::norm(F)){ //Is it a better Estimation (Less Cost)?
          auto newBaseLine0 =
          CostFunction::baseLine(newParams.at<double>(0, 0), newParams.at<double>(1, 0), params.at<double>(0, 1),
                                  params.at<double>(1, 1), params.at<double>(2, 1));
          auto newBaseLine1 =
          CostFunction::baseLine(newParams.at<double>(3, 0), newParams.at<double>(4, 0), params.at<double>(3, 1),
                                      params.at<double>(4, 1), params.at<double>(5, 1));

          params.at<double>(0, 0) = 0;                           // A0
          params.at<double>(1, 0) = 0;                           // B0
          params.at<double>(2, 0) = newParams.at<double>(2, 0);  // T0

          params.at<double>(0, 1) = newBaseLine0(0);  // X0
          params.at<double>(1, 1) = newBaseLine0(1);  // Y0
          params.at<double>(2, 1) = newBaseLine0(2);  // Z0

          params.at<double>(3, 0) = 0;                           // A1
          params.at<double>(4, 0) = 0;                           // B1
          params.at<double>(5, 0) = newParams.at<double>(5, 0);  // T2

          params.at<double>(3, 1) = newBaseLine1(0);  // X1
          params.at<double>(4, 1) = newBaseLine1(1);  // Y1
          params.at<double>(5, 1) = newBaseLine1(2);  // Z1
          ROS_INFO("accepted\r\n");
          lambda = 0.2 * lambda;
          break;
      }else{
        ROS_INFO_STREAM("receted lambda :" << lambda << std::endl);
        lambda = 10.0 * lambda;
      }

      ROS_INFO_STREAM("Inner Loop: " << delta << std::endl);
    }while(lambda >= 1.0 || cv::norm(delta, cv::NORM_INF) >= (thetha / 1000.0));

  }




void IterativeRefinement::refine(unsigned int n)
{

  _slidingWindow.exportMatlabData();
  ROS_INFO_STREAM("BEFORE: " << std::endl
                             << "st0: " << _slidingWindow.getPosition(0) << std::endl
                             << "st1: " << _slidingWindow.getPosition(1) << std::endl
                             << "st2: " << _slidingWindow.getPosition(2) << std::endl);

  assert(n == 3);  // TODO: currently only WindowSize 3 available
  RefinementData data;

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

  cv::Mat params(6, 2, CV_64F);

  params.at<double>(0, 0) = 0.0;                                                                                // A0
  params.at<double>(1, 0) = 0.0;                                                                                // B0
  params.at<double>(2, 0) = CostFunction::scale(1);                                                             // T0
  params.at<double>(0, 1) = u0(0);                                                                              // X0
  params.at<double>(1, 1) = u0(1);                                                                              // Y0
  params.at<double>(2, 1) = u0(2);                                                                              // Z0

  params.at<double>(3, 0) = 0.0;                                                                                // A1
  params.at<double>(4, 0) = 0.0;                                                                                // B1
  params.at<double>(5, 0) = CostFunction::scale(1);                                                             // T1
  params.at<double>(3, 1) = u1(0);                                                                              // X1
  params.at<double>(4, 1) = u1(1);                                                                              // Y1
  params.at<double>(5, 1) = u1(2);                                                                              // Z1

  data.R0 = _slidingWindow.getRotation(0);
  data.R1 = _slidingWindow.getRotation(1);
  data.R2 = _slidingWindow.getRotation(2);

  std::vector<std::vector<cv::Vec3d>*> vectors{ &(data.m0), &(data.m1), &(data.m2) };

  _slidingWindow.getCorrespondingFeatures(n - 1, 0, vectors);

  this->GaussNewton(data, params);

  n0 = CostFunction::scale(params.at<double>(2, 0));  // T0
  n1 = CostFunction::scale(params.at<double>(5, 0));  // T1
  u0 = cv::Vec3d(params.at<double>(0, 1), params.at<double>(1, 1), params.at<double>(2, 1));
  u1 = cv::Vec3d(params.at<double>(3, 1), params.at<double>(4, 1), params.at<double>(5, 1));

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

cv::Vec3d IterativeRefinement::CostFunction::baseLine(double a, double b, double x, double y, double z)
{

  cv::Vec3d v(x,y,z);

  cv::Matx33d A(
    1.0 - a*a , -2.0 * a,   0,
    2*a       , 1.0 - a*a,  0,
    0         , 0        ,  1 
  );

  cv::Matx33d B(
    1 - b*b   , 0        ,   2*b,
    0         , 1        ,    0,
    -2.0*b    , 0        ,  1.0 - b*b 
  );

  return A * B * v / ((1.0 + a*a) * (1.0 + b*b));
}



cv::Vec3d IterativeRefinement::CostFunction::baseLineDeriveA(double a, double b, double x, double y, double z)
{
  cv::Vec3d baseLine(
    2.0 * y * a + 4.0 * (b*b-1.0)*x*a-8.0*b*z*a-2*y,
    2.0*(a*a-1)*(b*b-1)*x-4.0*(b*z*a*a+y*a-b*z),
    2*a*(z*b*b+2*x*b-z)
  );
  return baseLine / std::pow((a*a + 1.0),2)*(b*b+1);
}

cv::Vec3d IterativeRefinement::CostFunction::baseLineDeriveB(double a, double b, double x, double y, double z)
{
  cv::Vec3d baseLine(
    2.0*(-z*a*a+2*b*((a*a-1.0)*x*a*y)+(a*a-1.0)*b*b*z+z),
    2.0*(b*y*a*a-2.0*(z*b*b+2*x*b-z)*a-b*y),
    2*(b*b-1)*x-4*b*z
  );
  return baseLine / (a*a+1.0)*std::pow((b*b+1.0),2);
}

double IterativeRefinement::CostFunction::scaleDeriveT(double t)
{
  return (std::exp(-1.0 * t) * (HIGH_VALUE - LOW_VALUE)) / std::pow(std::exp(-1.0 * t) + 1, 2);
}

double IterativeRefinement::CostFunction::scale(double t)
{
  return LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1 + std::exp(-1.0 * t)));
}

double IterativeRefinement::CostFunction::func(const cv::Vec3d & mk1, const cv::Matx33d & Rk1, const cv::Vec3d & uk, const cv::Matx33d & Rk, const cv::Vec3d & mk){
  return cv::Mat(mk1.t() * Rk1.t() * uk.cross(Rk * mk)).at<double>(0,0);
}

double IterativeRefinement::CostFunction::funcWhole(const cv::Vec3d & mk2, const cv::Matx33d & Rk2, double x1, double y1, double z1, const cv::Matx33d & Rk1, const cv::Vec3d & mk1, double x, double y, double z, const cv::Matx33d & Rk, const cv::Vec3d & mk, double a1, double b1, double t1, double a, double b, double t){
  double nk1 = scale(t1);
  double nk = scale (t);

  cv::Vec3d uk1 = baseLine(a1,b1, x1,y1,z1);
  cv::Vec3d uk = baseLine(a,b,x,y,z);
  
  
  cv::Vec3d uk20 = nk1 * uk1 + nk * uk;
  uk20 = uk20 / cv::norm(uk20);
  
  double cost = 
                std::pow(cv::Mat(mk2.t() * Rk2.t() * uk1.cross(Rk1 * mk1)).at<double>(0,0),2) +
                std::pow(cv::Mat(mk1.t() * Rk1.t() * uk.cross(Rk * mk)).at<double>(0,0),2) +
                std::pow(cv::Mat(mk2.t() * Rk2.t() * uk20.cross(Rk * mk)).at<double>(0,0),2);

return cost;
}