#include <opencv2/core.hpp>

class IterativeRefinement
{
private:
  double DERIV_STEP = 1e-5;
  double THRESHOLD = 1e-3;

public:
  IterativeRefinement();
  ~IterativeRefinement();

  struct Input
  {
    const cv::Mat mt;
    const cv::Mat Rt;
    const cv::Mat mhi;
    const cv::Mat Rhi;
    const cv::Mat shi;
  };

  double Func(const Input &input, const cv::Mat &st);
  double Deriv(double (*Func)(const Input &input, const cv::Mat &st), const Input &input, const cv::Mat &params,
                      int n);
  void GaussNewton(double (*Func)(const Input &input, const cv::Mat &st), const std::vector<Input> &inputs,
                          cv::Mat &params);
};
