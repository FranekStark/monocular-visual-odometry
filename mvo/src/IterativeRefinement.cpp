#include "IterativeRefinement.hpp"

#include <ros/ros.h>

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(){}
IterativeRefinement::~IterativeRefinement(){}

double IterativeRefinement::Func(const Input &input, const double & a, const double & b, const double & t)
{
  cv::Vec3d baseLine(1.0 - a*a - b*b, 2.0 * a, 2.0 * b);
  baseLine = baseLine / (1.0+a*a+b*b);
  double scale = LOW_VALUE + ((HIGH_VALUE - LOW_VALUE)/(1.0+std::exp(-1.0 * t)));

  return input.mt.dot(input.Rt.t() * (input.sign * scale * baseLine).cross(input.Rhi * input.mhi));
}

// double IterativeRefinement::DeriveA(const Input &input, const double & a, const double & b, const double & t){

// }

// double IterativeRefinement::DeriveB(const Input &input, const double & a, const double & b, const double & t){

// }

// double IterativeRefinement::DeriveT(const Input &input, const double & a, const double & b, const double & t){

// }

double IterativeRefinement::Deriv(const Input &input,
                                 const double & a, const double & b, const double & t, unsigned int n)
{
  // Assumes input is a single collumn cv::matrix

  double t1 = t;
  double t2 = t;
  double a1 = a;
  double a2 = a;
  double b1 = b;
  double b2 = b;

  switch (n)
  {
  case 0:
    a1 -= DERIV_STEP;
    a2 += DERIV_STEP;
    break;
  case 1:
    b1 -= DERIV_STEP;
    b2 += DERIV_STEP;
    break;
  case 2:
    t1 -= DERIV_STEP;
    t2 += DERIV_STEP;
    break;
  default:
    break;
  }


  double p1 = this->Func(input, a1, b1, t1);
  double p2 = this->Func(input, a2, b2, t2);

  double d = (p2 - p1) / (2 * DERIV_STEP);

  return d;
}

void IterativeRefinement::GaussNewton(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const double &sign, double & a, double &b, double & t)
{
  assert(mt.size() == mhi.size());
  int k = 2; //Three Params
  int n = mt.size();
  cv::Mat delta;

  cv::Mat J(n, k, CV_64F);  // Jacobian of Func()
  cv::Mat f(n, 1, CV_64F);  // f

  do
  {
    /**
     * Generate Jacobi and F
     */
    // at(row, col)
    for (int i = 0; i < n; i++)
    {
      const Input input{
        mt[i],
        Rt,
        mhi[i],
        Rhi,
        sign
      };

      f.at<double>(i, 0) = (this->Func(input, a,b,t));
      for (int j = 0; j < k; j++)
      {
        J.at<double>(i, j) = Deriv(input, a,b,t,j);
      }
    }

    //J'*J * d = -J'*f 

    cv::solve(J.t()*J, (-1 * J.t()) * f, delta, cv::DECOMP_NORMAL);
    //ROS_INFO_STREAM("delta: " << delta << std::endl);
    //delta = -(J.t()+J).inv()*J.t()*f;
    a += delta.at<double>(0,0);
    b += delta.at<double>(1,0);
    //t += delta.at<double>(2,0);
    ROS_INFO_STREAM("delta: " << delta << std::endl);
  } while (cv::norm(delta) > THRESHOLD);
}

void IterativeRefinement::iterativeRefinement(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d & st, const double & sign){
  cv::Vec3d baseLine = st - shi;
  double scale = cv::norm(baseLine, cv::NORM_L2);
  scale =1;
  baseLine = baseLine / cv::norm(baseLine, cv::NORM_L2);
  double x = baseLine(0);
  double y = baseLine(1);
  double z = baseLine(2); 
  double a,b,t;
  ROS_INFO_STREAM("Scale: " << scale << std::endl);
  t = -1.0 * std::log((HIGH_VALUE-scale) / (scale-LOW_VALUE));
  if(x >= 0){ //TODO, wer nimmt den null Fall?
    a = y / (std::sqrt(-(y*y)-(z*z)+1)+1);
    b = z / (std::sqrt(-(y*y)-(z*z)+1)+1);
    ROS_ERROR_STREAM_COND(std::abs(x - std::sqrt(-(y*y)-(z*z)+1)) > 0.01, "Vektor konnte nicht parametriesiert werden: " << baseLine << std::endl);
  }else{
    a = -1.0 * y / (std::sqrt(-(y*y)-(z*z)+1)-1);
    b = -1.0 * z / (std::sqrt(-(y*y)-(z*z)+1)-1);
    ROS_ERROR_STREAM_COND(std::abs(x - (-1*std::sqrt(-(y*y)-(z*z)+1))) > 0.01, "Vektor konnte nicht parametriesiert werden: " << baseLine << std::endl);
  }
  
  ROS_INFO_STREAM("Before: a: " << a << ", b: " << b << ", t: " << t << std::endl);

  this->GaussNewton(mt,Rt,mhi,Rhi,sign, a,b,t);

  ROS_INFO_STREAM("Refined: a: " << a << ", b: " << b << ", t: " << t << std::endl);

  /*Calc BaseLine */
  x = 1.0 - a*a - b*b;
  y = 2.0 * a;
  z = 2.0 * b;

  baseLine(0) = x;
  baseLine(1) = y;
  baseLine(2) = z;

  baseLine = baseLine / (1.0 + a*a + b*b);
  
  scale = LOW_VALUE + ((HIGH_VALUE-LOW_VALUE) / (1+std::exp(-t)));

  baseLine = baseLine * scale;

  st = shi + sign * baseLine;

}
