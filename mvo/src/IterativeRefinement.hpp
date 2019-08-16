#include <opencv2/core.hpp>
#include "SlidingWindow.hpp"

class IterativeRefinement
{
private:
  SlidingWindow & _slidingWindow;

  double THRESHOLD = 0.0001;

  struct RefinementData
  {
    std::vector<cv::Vec3d>  m2;
    std::vector<cv::Vec3d>  m1;
    std::vector<cv::Vec3d>  m0;
    cv::Matx33d  R2;
    cv::Matx33d  R1;
    cv::Matx33d  R0;
  };
 
  class CostFunction
  {
     public:

    static constexpr double HIGH_VALUE = 10;
    static constexpr double LOW_VALUE = 0.25;
    static constexpr double DERIV_STEP = std::sqrt(DBL_EPSILON);

    struct Input
    {
      const cv::Vec3d & m2;
      const cv::Vec3d & m1;
      const cv::Vec3d & m0;
      const cv::Matx33d & R2;
      const cv::Matx33d & R1;
      const cv::Matx33d & R0;
    };

    static double func10(const Input & input, double a, double b, double x, double y, double z);
    static double func21(const Input & input, double a1, double b1, double x1, double y1, double z1);
    static double func20(const Input & input, double a, double b, double x, double y, double z, double t, double a1, double b1, double x1, double y1, double z1, double t1);

    static double func(const cv::Vec3d & mk1, const cv::Matx33d & Rk1, const cv::Vec3d & uk, const cv::Matx33d & Rk, const cv::Vec3d & mk);
    static double derive(const Input & input, const cv::Mat & params, unsigned int index);
    static cv::Vec3d baseLine(double a, double b, double x, double y, double z);
    static cv::Vec3d baseLineDeriveA(double a, double b, double x, double y, double z);
    static cv::Vec3d baseLineDeriveB(double a, double b, double x, double y, double z);
    static double scale(double t);
    static double scaleDeriveT(double t);

  };
 
 void CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const RefinementData & data, const cv::Mat & params);
 cv::Mat CreateFunction(const RefinementData & data, const cv::Mat & params);
 void GaussNewton(const RefinementData & data, cv::Mat & params);

 static double derivationParam(double x);


public:
  IterativeRefinement(SlidingWindow & slidingWindow);
  ~IterativeRefinement();

  void refine(unsigned int n);
  

};
