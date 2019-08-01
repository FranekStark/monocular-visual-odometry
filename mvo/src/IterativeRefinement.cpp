#include "IterativeRefinement.hpp"

#include <ros/ros.h>

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(){}
IterativeRefinement::~IterativeRefinement(){}

double IterativeRefinement::Func(const Input &input, const cv::Vec3d &st)
{
  // auto b = st - input.shi;
  // auto Rhimhi = input.Rhi * input.mhi;
  // auto bCross = b.cross(Rhimhi);
  // auto RtCross = input.Rt.t() * bCross;
  // return input.mt.dot(RtCross);
  return input.mt.dot(input.Rt.t() * (st - input.shi).cross(input.Rhi * input.mhi));
}

double IterativeRefinement::Deriv(const Input &input,
                                  const cv::Vec3d &params, int n)
{
  // Assumes input is a single collumn cv::matrix

  // Returns the derivative of the nth parameter (bx, by or bz)
  cv::Vec3d params1(params);
  cv::Vec3d params2(params);

  // Use central difference  to get derivative
  params1(n) -= DERIV_STEP;
  params2(n) += DERIV_STEP;

  double p1 = this->Func(input, params1);
  double p2 = this->Func(input, params2);

  double d = (p2 - p1) / (2 * DERIV_STEP);

  return d;
}

void IterativeRefinement::GaussNewton(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d &params)
{
  assert(mt.size() == mhi.size());
  int k = params.rows;  // Should be 3
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
        shi
      };

      f.at<double>(i, 0) = (this->Func(input, params));
      for (int j = 0; j < k; j++)
      {
        J.at<double>(i, j) = Deriv(input, params, j);
      }
    }

    //J'*J * d = -J'*f 

    cv::solve(J.t()*J, (-1 * J.t()) * f, delta, cv::DECOMP_NORMAL);
    //ROS_INFO_STREAM("delta: " << delta << std::endl);
    params(0) += delta.at<double>(0,0);
    params(1) += delta.at<double>(1,0);
    params(2) += delta.at<double>(2,0);
  } while (cv::norm(delta) > THRESHOLD);
}

void IterativeRefinement::iterativeRefinement(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d & st){
  this->GaussNewton(mt,Rt, mhi, Rhi, shi, st);
}
