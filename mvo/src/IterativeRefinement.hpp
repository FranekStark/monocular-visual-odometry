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
    static constexpr double DERIV_STEP = 1e-5;

    struct Input
    {
      const cv::Vec3d & m2;
      const cv::Vec3d & m1;
      const cv::Vec3d & m0;
      const cv::Matx33d & R2;
      const cv::Matx33d & R1;
      const cv::Matx33d & R0;
    };

    static double func(const Input & input, const cv::Mat & params);
    static double derive(const Input & input, const cv::Mat & params, unsigned int index);
    static cv::Vec3d baseLine(double a, double b, double x, double y, double z);
    static double scale(double t);

  };
 
 void CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const RefinementData & data, const cv::Mat & params);
 cv::Mat CreateFunction(const RefinementData & data, const cv::Mat & params);
 void GaussNewton(const RefinementData & data, cv::Mat & params);


public:
  IterativeRefinement(SlidingWindow & slidingWindow);
  ~IterativeRefinement();

  void refine(unsigned int n);
  

};
