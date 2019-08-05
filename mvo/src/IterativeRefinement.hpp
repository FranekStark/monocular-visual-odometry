#include <opencv2/core.hpp>

class IterativeRefinement
{
private:
  double DERIV_STEP = 1e-5;
  double THRESHOLD = 1e-3;
  double HIGH_VALUE = 2;
  double LOW_VALUE = 0.25;

  struct Input
  {
    const cv::Vec3d & mt;
    const cv::Matx33d & Rt;
    const cv::Vec3d & mhi;
    const cv::Matx33d & Rhi;
    const double & sign;
  };


  double Func(const Input &input, const double & a, const double & b, const double & t);
  // double DeriveA(const Input &input, const double & a, const double & b, const double & t);
  // double DeriveB(const Input &input, const double & a, const double & b, const double & t);
  // double DeriveT(const Input &input, const double & a, const double & b, const double & t);

  double Deriv(const Input &input, const double & a, const double & b, const double & t, unsigned int n);
  void GaussNewton(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const double &sign, double & a, double &b, double & t);

public:
  IterativeRefinement();
  ~IterativeRefinement();

  void iterativeRefinement(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d & st,const double & sign);

  

};
