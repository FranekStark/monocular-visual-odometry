#include "IterativeRefinement.hpp"

#include <ros/ros.h>

#include <limits>

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(SlidingWindow slidingWindow): _slidingWindow(slidingWindow)
{
}
IterativeRefinement::~IterativeRefinement()
{
}

cv::Vec3d IterativeRefinement::CalculateEstimatedBaseLine(const double &a, const double &b, const double &x,
                                                          const double &y, const double &z)
{
  cv::Vec3d baseLine(1.0 - a * a - b * b, 2.0 * a, 2.0 * b);
  baseLine = baseLine / (1.0 + a * a + b * b);
  cv::Vec3d baseLineL;
                                                                   //TODO: faste Computation!
  baseLineL(0) = x * baseLine(0) - y * baseLine(1) - z * baseLine(2);
  baseLineL(1) = x * baseLine(1) + y * baseLine(0);
  baseLineL(2) = x * baseLine(2) + z * baseLine(0);
  return baseLineL;
}

double IterativeRefinement::Func(const Input &input, const double &a, const double &b, const double &t)
{
  cv::Vec3d baseLine = this->CalculateEstimatedBaseLine(a, b, input.xBefore, input.yBefore, input.zBefore);

  double scale = LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1.0 + std::exp(-1.0 * t)));

  return input.mt.dot(input.Rt.t() * (input.sign * scale * baseLine).cross(input.Rhi * input.mhi));
}

double IterativeRefinement::DeriveA(const Input &input, const double &a, const double &b, const double &t)
{
  cv::Vec3d baseLine(1.0 - a * a - b * b, 2.0 * a, 2.0 * b);
  cv::Vec3d baseLineDerive(-2.0 * a, 2.0, 0);

  double baseLineNormer = 1.0 / (1.0 + a * a + b * b);
  double baseLineDerivNormer = (-2.0 * a) / ((a * a + b * b + 1) * (a * a + b * b + 1));
  double scale = LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1.0 + std::exp(-1.0 * t)));

  return input.mt.dot(input.Rt.t() *
                      (input.sign * scale * (baseLineDerive * baseLineNormer + baseLine * baseLineDerivNormer))
                          .cross(input.Rhi * input.mhi));
}

double IterativeRefinement::DeriveB(const Input &input, const double &a, const double &b, const double &t)
{
  cv::Vec3d baseLine(1.0 - a * a - b * b, 2.0 * a, 2.0 * b);
  cv::Vec3d baseLineDerive(-2.0 * b, 0, 2.0);

  double baseLineNormer = 1.0 / (1.0 + a * a + b * b);
  double baseLineDerivNormer = (-2.0 * b) / ((a * a + b * b + 1) * (a * a + b * b + 1));
  double scale = LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1.0 + std::exp(-1.0 * t)));

  return input.mt.dot(input.Rt.t() *
                      (input.sign * scale * (baseLineDerive * baseLineNormer + baseLine * baseLineDerivNormer))
                          .cross(input.Rhi * input.mhi));
}

double IterativeRefinement::DeriveT(const Input &input, const double &a, const double &b, const double &t)
{
  cv::Vec3d baseLine(1.0 - a * a - b * b, 2.0 * a, 2.0 * b);
  double baseLineNormer = 1.0 / (1.0 + a * a + b * b);

  double scale = std::exp(-1.0 * t) * (HIGH_VALUE - LOW_VALUE) / std::pow((1.0 + std::exp(-1.0 * t)), 2);

  return input.mt.dot(input.Rt.t() * (input.sign * scale * (baseLineNormer * baseLine)).cross(input.Rhi * input.mhi));
}

double IterativeRefinement::Deriv(const Input &input, const double &a, const double &b, const double &t, unsigned int n)
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

void IterativeRefinement::CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const std::vector<cv::Vec3d> &mt,
                                                    const cv::Matx33d &Rt, const std::vector<cv::Vec3d> &mhi,
                                                    const cv::Matx33d &Rhi, const double &sign, const double &a,
                                                    const double &b, const double &t, const double &x, const double &y,
                                                    const double &z)
{
  assert(mt.size() == mhi.size());
  int n = mt.size();
  for (int i = 0; i < n; i++)
  {
    const Input input{ mt[i], Rt, mhi[i], Rhi, sign, x, y, z };

    F.at<double>(i, 0) = (this->Func(input, a, b, t));
    J.at<double>(i, 0) = this->Deriv(input, a, b, t, 0);
    J.at<double>(i, 1) = this->Deriv(input, a, b, t, 1);
    J.at<double>(i, 2) = this->Deriv(input, a, b, t, 2);
  }
}

cv::Mat IterativeRefinement::CreateFunction(const std::vector<cv::Vec3d> &mt, const cv::Matx33d &Rt,
                                            const std::vector<cv::Vec3d> &mhi, const cv::Matx33d &Rhi,
                                            const double &sign, const double &a, const double &b, const double &t,
                                            const double &x, const double &y, const double &z)

{
  assert(mt.size() == mhi.size());
  int n = mt.size();
  cv::Mat f(n, 1, CV_64F);
  for (int i = 0; i < n; i++)
  {
    const Input input{ mt[i], Rt, mhi[i], Rhi, sign, x, y, z };

    f.at<double>(i, 0) = (this->Func(input, a, b, t));
  }

  return f;
}

void IterativeRefinement::CreateMultiJacobianAndFunction(cv::Mat J, cv::Mat F, const std::vector<RefinementData> & data){
  unsigned int nSamplesBefore = 0;
  for(unsigned int i = 0; i < data.size(); i++){ //TODO: Faster Access
    unsigned int nSamples = data[i].mhi.size();
    double sign = 1.0; //TODO: fake sign
    this->CreateJacobianAndFunction(J.colRange(i*3, (i*3)+3).rowRange(nSamplesBefore, nSamplesBefore + nSamples), F.rowRange(nSamplesBefore, nSamplesBefore + nSamples), 
    data[i].mt, data[i].Rt, data[i].mhi, data[i].Rhi, sign, data[i].a, data[i].b, data[i].t, data[i].x, data[i].y, data[i].z);
    nSamplesBefore += nSamples;
  }
}
cv::Mat IterativeRefinement::CreateMultiFunction(const std::vector<RefinementData> & data, cv::Mat newParams){
  cv::Mat f(0,1,CV_64F);
  for(unsigned int i = 0; i < data.size(); i++){ //TODO: Faster Access
    double sign = 1.0; //TODO: fake sign
    f.push_back(this->CreateFunction(data[i].mt, data[i].Rt, data[i].mhi, data[i].Rhi, sign, newParams.at<double>(3 * i + 0),newParams.at<double>(3*i + 1),newParams.at<double>(3 * i + 2), data[i].x, data[i].y, data[i].z));

  }
}

cv::Mat IterativeRefinement::GetAllParamsValues(std::vector<RefinementData> & data){
  std::vector<double> params;
  for(auto d: data){
    params.push_back(d.a);
    params.push_back(d.b);
    params.push_back(d.t);
  }

  return cv::Mat(params);
}

/*Isn't Gauss-Newotn! -> It's now Levemberg Marquardt... */
// TODO: From https://www.mrpt.org/Levenberg-Marquardt_algorithm
void IterativeRefinement::GaussNewton(std::vector<RefinementData> & data)
{
  /*params */
  double tau = 10E-3;
  double epsilon1, epsilon2, epsilon3, epsilon4;
  epsilon1 = epsilon2 = epsilon3 = 10E-12;
  epsilon4 = 0;
  unsigned int kmax = 100;

  cv::Mat delta;

  unsigned int n;
  for(auto d : data){
    n += d.mhi.size();
  }

  cv::Mat J(n, 3 * data.size(), CV_64F);  // Jacobian of Func()
  cv::Mat f(n, 1, CV_64F);  // f
  
  this->CreateMultiJacobianAndFunction(J,f,data);

  cv::Mat gradient = J.t() * f;
  cv::Mat A = J.t() * J;
  double mue;
  cv::minMaxLoc(A.diag(), NULL, &mue);
  mue = mue * tau;

  bool stop = (cv::norm(gradient, cv::NormTypes::NORM_INF) <= epsilon1);
  unsigned int k = 0;
  int v = 2;
  while (!stop && (k < kmax))
  {
    k++;
    double rho = 0;
    do
    {
      cv::solve(A + mue * cv::Mat::eye(3, 3, CV_64F), gradient, delta, cv::DECOMP_QR);
      if (cv::norm(delta, cv::NormTypes::NORM_L2) <=
          epsilon2 * (cv::norm(this->GetAllParamsValues(data), cv::NormTypes::NORM_L2) + epsilon2))
      {
        stop = true;
      }
      else
      {
        cv::Mat params = this->GetAllParamsValues(data);
        cv::Mat newParams = params + delta;
        cv::Mat fNew = this->CreateMultiFunction(data, newParams);

        rho = (cv::norm(f, cv::NormTypes::NORM_L2SQR) - cv::norm(fNew, cv::NormTypes::NORM_L2SQR)) /
              (0.5 * cv::Mat(delta.t() * ((mue * delta) - gradient)).at<double>(0, 0));
        if (rho > 0)
        {
          stop = ((cv::norm(f, cv::NormTypes::NORM_L2) - cv::norm(fNew, cv::NormTypes::NORM_L2)) <
                  (epsilon4 * cv::norm(f, cv::NormTypes::NORM_L2)));


          //TODO: entkoppel params
          for(unsigned int i = 0; i < data.size(); i++){
            auto newBaseLineEstimation = this->CalculateEstimatedBaseLine(newParams.at<double>(3 * i + 0), newParams.at<double>(3 * i + 1), data[i].x, data[i].y, data[i].z);
            data[i].a = 0;
            data[i].b = 0;
            data[i].t = newParams.at<double>(3 * i + 2);
            data[i].x = newBaseLineEstimation(0);
            data[i].y = newBaseLineEstimation(1);
            data[i].z = newBaseLineEstimation(2);
          }
          
          


          this->CreateMultiJacobianAndFunction(J,f,data);
          // ROS_INFO_STREAM("J: " << J << std::endl << "f: " << f << std::endl);
          gradient = J.t() * f;
          A = J.t() * J;

          stop = stop || (cv::norm(gradient, cv::NormTypes::NORM_INF) <= epsilon1);
          mue = mue * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1, 3));
          v = 2;
        }
        else
        {
          mue = mue * v;
          v = 2 * v;
        }
      }
    } while (rho <= 0 && !stop);
    stop = (cv::norm(f, cv::NormTypes::NORM_L2) <= epsilon3);
  }
}

void IterativeRefinement::iterativeRefinement(const std::vector<cv::Vec3d> &mt, const cv::Matx33d &Rt,
                                              const std::vector<cv::Vec3d> &mhi, const cv::Matx33d &Rhi,
                                              const cv::Vec3d &shi, cv::Vec3d &st, const double &sign)
{
  cv::Vec3d baseLine = st - shi;
  double scale = cv::norm(baseLine, cv::NORM_L2);
  scale = 1;
  baseLine = baseLine / cv::norm(baseLine, cv::NORM_L2);
  double x = baseLine(0);
  double y = baseLine(1);
  double z = baseLine(2);
  double a, b, t;
  a = b = 0;
  ROS_INFO_STREAM("Scale: " << scale << std::endl);
  t = -1.0 * std::log((HIGH_VALUE - scale) / (scale - LOW_VALUE));
  // if (x >= 0)
  // {  // TODO, wer nimmt den null Fall?
  //   a = y / (std::sqrt(-(y * y) - (z * z) + 1) + 1);
  //   b = z / (std::sqrt(-(y * y) - (z * z) + 1) + 1);
  //   ROS_ERROR_STREAM_COND(std::abs(x - std::sqrt(-(y * y) - (z * z) + 1)) > 0.01,
  //                         "Vektor konnte nicht parametriesiert werden: " << baseLine << std::endl);
  // }
  // else
  // {
  //   a = -1.0 * y / (std::sqrt(-(y * y) - (z * z) + 1) - 1);
  //   b = -1.0 * z / (std::sqrt(-(y * y) - (z * z) + 1) - 1);
  //   ROS_ERROR_STREAM_COND(std::abs(x - (-1 * std::sqrt(-(y * y) - (z * z) + 1))) > 0.01,
  //                         "Vektor konnte nicht parametriesiert werden: " << baseLine << std::endl);
  // }

  // ROS_INFO_STREAM("Before: a: " << a << ", b: " << b << ", t: " << t << std::endl);
  ROS_INFO_STREAM("Before BaseLine(x,y,z): " << baseLine << std::endl);
  ROS_INFO_STREAM("Before Scale(t): " << scale  << std::endl);
  // ROS_INFO_STREAM("-> Before Base_Line(a,b): "
  //                 << cv::Vec3d(1.0 - a * a - b * b, 2.0 * a, 2.0 * b) / (1.0 + a * a + b * b) << std::endl);
  // ROS_INFO_STREAM("-> Before Scale(a,b): " << LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1 + std::exp(-t))) <<
  // std::endl);

  this->GaussNewton(mt, Rt, mhi, Rhi, sign, a, b, t, x, y, z);

  // ROS_INFO_STREAM("Refined: a: " << a << ", b: " << b << ", t: " << t << std::endl);
  // ROS_INFO_STREAM("-> After Base_Line(a,b): "
  //                 << cv::Vec3d(1.0 - a * a - b * b, 2.0 * a, 2.0 * b) / (1.0 + a * a + b * b) << std::endl);
  // ROS_INFO_STREAM("-> After Scale(a,b): " << LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1 + std::exp(-t))) <<
  // std::endl);

  // /*Calc BaseLine */
  // x = 1.0 - a * a - b * b;
  // y = 2.0 * a;
  // z = 2.0 * b;
  baseLine(0) = x;
  baseLine(1) = y;
  baseLine(2) = z;

  // baseLine = baseLine / (1.0 + a * a + b * b);

  scale = LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1 + std::exp(-t)));
  ROS_INFO_STREAM("After BaseLine(x,y,z): " << baseLine << std::endl);
  ROS_INFO_STREAM("Afer Scale(t): " << scale  << std::endl);
  baseLine = baseLine * scale;
  
  st = shi + sign * baseLine;
}

void IterativeRefinement::refine(unsigned int n){
  std::vector<RefinementData> data(n);

  for(unsigned int i = 0; i < n; i++){
    cv::Vec3d & stNow = _slidingWindow.getPosition(i);
    cv::Vec3d & stBefore = _slidingWindow.getPosition(i+1);
    double norm = cv::norm(stNow - stBefore, cv::NormTypes::NORM_L2);
    cv::Vec3d baseLine = (stNow - stBefore) / norm; 
    data[i].a = 0;
    data[i].b = 0;
    data[i].t = -1.0 * std::log((HIGH_VALUE - norm) / (norm - LOW_VALUE));
    data[i].x = baseLine(0);
    data[i].y = baseLine(1);
    data[i].z = baseLine(2);
    data[i].Rhi = _slidingWindow.getRotation(n+1);
    data[i].Rt = _slidingWindow.getRotation(n);
    _slidingWindow.getCorrespondingFeatures(n+1,n, data[i].mhi, data[i].mt);
  }



  
}
