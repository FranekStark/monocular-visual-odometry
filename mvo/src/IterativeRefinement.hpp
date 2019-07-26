#include <opencv2/core.hpp>

class IterativeRefinement
{
private:
  double DERIV_STEP = 1e-5;
  double THRESHOLD = 1e-3;

  struct Input
  {
    const cv::Vec3d & mt;
    const cv::Matx33d & Rt;
    const cv::Vec3d & mhi;
    const cv::Matx33d & Rhi;
    const cv::Vec3d & shi;
  };


  double Func(const Input &input, const cv::Vec3d &st);
  double Deriv(const Input &input, const cv::Vec3d &params, int n);
  void GaussNewton(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi,
                          cv::Vec3d &params);

public:
  IterativeRefinement();
  ~IterativeRefinement();

  void iterativeRefinement(const std::vector<cv::Vec3d> & mt, const cv::Matx33d & Rt, const std::vector<cv::Vec3d> & mhi, const cv::Matx33d & Rhi, const cv::Vec3d & shi, cv::Vec3d & st);

  

};
