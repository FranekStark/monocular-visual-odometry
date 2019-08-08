#include <opencv2/core.hpp>
#include "SlidingWindow.hpp"

class IterativeRefinement
{
private:
  SlidingWindow & _slidingWindow;
  double DERIV_STEP = 1e-5;
  double THRESHOLD = 0.0001;
  double HIGH_VALUE = 10;
  double LOW_VALUE = 0.25;

  struct Input
  {
    const cv::Vec3d & mt;
    const cv::Matx33d & Rt;
    const cv::Vec3d & mhi;
    const cv::Matx33d & Rhi;
    const double & sign;
    const double xBefore;
    const double yBefore;
    const double zBefore;
  };

  struct RefinementData
  {
    std::vector<cv::Vec3d>  mt;
    cv::Matx33d  Rt;
    std::vector<cv::Vec3d>  mhi;
    cv::Matx33d  Rhi;
    double x;
    double y;
    double z;
    double a;
    double b;
    double t;
  };
 


  double Func(const Input &input, const double & a, const double & b, const double & t);
  double DeriveA(const Input &input, const double & a, const double & b, const double & t);
  double DeriveB(const Input &input, const double & a, const double & b, const double & t);
  double DeriveT(const Input &input, const double & a, const double & b, const double & t);


 void CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const double &sign, const double & a, const double & b, const double & t,const double &x, const double &y, const double &z);
 cv::Mat CreateFunction(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const double &sign, const double & a, const double & b, const double & t,const double &x, const double &y, const double &z);
 void CreateMultiJacobianAndFunction(cv::Mat J, cv::Mat F, const std::vector<RefinementData> & data);
 cv::Mat CreateMultiFunction(const std::vector<RefinementData> & data, cv::Mat newParams);
 double Deriv(const Input &input, const double & a, const double & b, const double & t, unsigned int n);
 void GaussNewton(std::vector<RefinementData> & data);

  cv::Vec3d CalculateEstimatedBaseLine(const double & a, const double & b, const double &x, const double &y, const double &z);

  cv::Mat GetAllParamsValues(std::vector<RefinementData> & data);

public:
  IterativeRefinement(SlidingWindow & slidingWindow);
  ~IterativeRefinement();

  void iterativeRefinement(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d & st,const double & sign);
  void refine(unsigned int n);
  

};
