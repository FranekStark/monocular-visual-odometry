#include "IterativeRefinement.hpp"

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(){};
IterativeRefinement::~IterativeRefinement(){};

double IterativeRefinement::Func(const Input &input, const cv::Mat &st)
{
  auto b = st - input.shi;
  auto Rhimhi = input.Rhi * input.mhi;
  auto bCross = b.cross(Rhimhi);
  auto RtCross = input.Rt.t() * bCross;
  return input.mt.dot(RtCross);
  // return cv::Mat(input.mt * input.Rt.t() * (st - input.shi).cross((input.Rhi * input.mhi))).at<double>(0,0);
}

double IterativeRefinement::Deriv(double (*Func)(const Input &input, const cv::Mat &st), const Input &input,
                                  const cv::Mat &params, int n)
{
  // Assumes input is a single collumn cv::matrix

  // Returns the derivative of the nth parameter (bx, by or bz)
  cv::Mat params1 = params.clone();
  cv::Mat params2 = params.clone();

  // Use central difference  to get derivative
  params1.at<double>(n, 0) -= DERIV_STEP;
  params2.at<double>(n, 0) += DERIV_STEP;

  double p1 = Func(input, params1);
  double p2 = Func(input, params2);

  double d = (p2 - p1) / (2 * DERIV_STEP);

  return d;
}

void IterativeRefinement::GaussNewton(double (*Func)(const Input &input, const cv::Mat &st),
                                      const std::vector<Input> &inputs, cv::Mat &params)
{
  int k = params.rows;  // Should be 3
  int n = inputs.size();
  cv::Mat delta(3, 1, CV_64F);

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
      const Input &input = inputs[i];
      f.at<double>(i, 1) = -Func(input, params);
      for (int j = 0; j < k; j++)
      {
        J.at<double>(i, j) = Deriv(Func, input, params, j);
      }
    }

    // J * delta = f;
    cv::solve(J, f, delta, cv::DECOMP_NORMAL);
    params += delta;
  } while (cv::norm(delta) > THRESHOLD);
}