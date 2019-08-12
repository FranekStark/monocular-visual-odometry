#include "IterativeRefinement.hpp"

#include <ros/ros.h>

#include <limits>

// TODO: FROM: https://nghiaho.com/?page_id=355

IterativeRefinement::IterativeRefinement(SlidingWindow & slidingWindow): _slidingWindow(slidingWindow)
{
}
IterativeRefinement::~IterativeRefinement()
{
}




void IterativeRefinement::CreateJacobianAndFunction(cv::Mat J, cv::Mat F, const RefinementData & data, const cv::Mat & params){
  assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());
  for(unsigned int i = 0; i < data.m0.size(); i++){ //TODO: Faster Access
    CostFunction::Input input {
      data.m2[i],
      data.m1[i],
      data.m0[i],
      data.R2,
      data.R1,
      data.R0
    };

    double errValue = CostFunction::func(input, params);
    double deriveA0 = CostFunction::derive(input, params, 0);
    double deriveB0 = CostFunction::derive(input, params, 1);
    double deriveT0 = CostFunction::derive(input, params, 2);
    double deriveA1 = CostFunction::derive(input, params, 3);
    double deriveB1 = CostFunction::derive(input, params, 4);
    double deriveT1 = CostFunction::derive(input, params, 5);
    
    F.at<double>(i,0) = errValue; //TODO: inittlegnterere Zugriff!!!
    J.at<double>(i,0) = deriveA0;
    J.at<double>(i,1) = deriveB0;
    J.at<double>(i,2) = deriveT0;
    J.at<double>(i,3) = deriveA1;
    J.at<double>(i,4) = deriveB1;
    J.at<double>(i,5) = deriveT1;
  }
}

cv::Mat IterativeRefinement::CreateFunction(const RefinementData & data, const cv::Mat & params){
  assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());
  cv::Mat f(data.m0.size(),1,CV_64F);
  for(unsigned int i = 0; i < data.m1.size(); i++){ //TODO: Faster Access
    CostFunction::Input input {
      data.m2[i],
      data.m1[i],
      data.m0[i],
      data.R2,
      data.R1,
      data.R0
    };
    
    double value = CostFunction::func(input, params);
    f.at<double>(i,0) = value;
  }
  return f;
}

/*Isn't Gauss-Newotn! -> It's now Levemberg Marquardt... */
// TODO: From https://www.mrpt.org/Levenberg-Marquardt_algorithm
void IterativeRefinement::GaussNewton(const RefinementData & data, cv::Mat & params)
{
  assert(data.m0.size() == data.m1.size() && data.m1.size() == data.m2.size());

  /*params */
  double tau = 10E-3;
  double epsilon1, epsilon2, epsilon3, epsilon4;
  epsilon1 = epsilon2 = epsilon3 = 10E-12;
  epsilon4 = 0;
  unsigned int kmax = 100;

  cv::Mat delta(6, 1, CV_64F);

  unsigned int n = data.m0.size();

  cv::Mat J(n, 6, CV_64F);  // Jacobian of Func()
  cv::Mat f(n, 1, CV_64F);  // f
  
  this->CreateJacobianAndFunction(J, f, data, params);

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
      cv::solve(A + mue * cv::Mat::eye(A.rows, A.cols, CV_64F), gradient, delta, cv::DECOMP_QR);
      if (cv::norm(delta, cv::NormTypes::NORM_L2) <=
          epsilon2 * (cv::norm(params.col(0), cv::NormTypes::NORM_L2) + epsilon2))
      {
        stop = true;
      }
      else
      {
        cv::Mat newParams = params.clone();
        newParams.col(0) += delta;
        cv::Mat fNew = this->CreateFunction(data, newParams);

        rho = (cv::norm(f, cv::NormTypes::NORM_L2SQR) - cv::norm(fNew, cv::NormTypes::NORM_L2SQR)) /
              (0.5 * cv::Mat(delta.t() * ((mue * delta) - gradient)).at<double>(0, 0));
              ROS_INFO_STREAM("f " << f << std::endl);
              ROS_INFO_STREAM("fNew " << fNew << std::endl);
              ROS_INFO_STREAM("rho " << rho << std::endl);
        if (rho > 0)
        {
          stop = ((cv::norm(f, cv::NormTypes::NORM_L2) - cv::norm(fNew, cv::NormTypes::NORM_L2)) <
                  (epsilon4 * cv::norm(f, cv::NormTypes::NORM_L2)));

            //TODO: bessere Zugriffe auf Mat
            auto newBaseLine0 = CostFunction::baseLine(newParams.at<double>(0,0),newParams.at<double>(1,0),params.at<double>(0,1),params.at<double>(1,1),params.at<double>(2,1));
            auto newBaseLine1 = CostFunction::baseLine(newParams.at<double>(3,0),newParams.at<double>(4,0),params.at<double>(3,1),params.at<double>(4,1),params.at<double>(5,1));
            
            params.at<double>(0,0) = 0; //A0
            params.at<double>(1,0) = 0; //B0
            params.at<double>(2,0) = newParams.at<double>(2,0); //T0

            params.at<double>(0,1) = newBaseLine0(0); //X0
            params.at<double>(1,1) = newBaseLine0(1); //Y0
            params.at<double>(2,1) = newBaseLine0(2); //Z0


            params.at<double>(3,0) = 0; //A1
            params.at<double>(4,0) = 0; //B1
            params.at<double>(5,0) = newParams.at<double>(5,0); //T2
        
            params.at<double>(3,1) = newBaseLine1(0); //X1
            params.at<double>(4,1) = newBaseLine1(1); //Y1
            params.at<double>(5,1) = newBaseLine1(2); //Z1

            ROS_INFO_STREAM("new Params set delta: " << std::endl << delta << std::endl <<"newParams: " << std::endl << newParams << std::endl);
            ROS_INFO_STREAM("params: " << std::endl << params << std::endl);
            ROS_INFO_STREAM("newBase0: " << newBaseLine0 << " norm:" << cv::norm(newBaseLine0) << std::endl);
            ROS_INFO_STREAM("newBase1: " << newBaseLine1 << " norm:" << cv::norm(newBaseLine1) << std::endl);

            this->CreateJacobianAndFunction(J,f,data,params);

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


void IterativeRefinement::refine(unsigned int n){
  ROS_INFO_STREAM( "BEFORE: " << std::endl <<
                  "st0: " << _slidingWindow.getPosition(0) << std::endl <<
                  "st1: " << _slidingWindow.getPosition(1) << std::endl <<
                  "st2: " << _slidingWindow.getPosition(2) << std::endl);

  assert(n == 3); //TODO: currently only WindowSize 3 available
  RefinementData data;

  cv::Vec3d & st0 = _slidingWindow.getPosition(0);
  cv::Vec3d & st1 = _slidingWindow.getPosition(1);
  cv::Vec3d & st2 = _slidingWindow.getPosition(2);

  double n0 = cv::norm(st0 - st1);
  double n1 = cv::norm(st1 - st2);
  cv::Vec3d u0 = (st0 - st1) / n0;
  cv::Vec3d u1 = (st1 - st2) / n1;

  ROS_INFO_STREAM("Before: " << std::endl);
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * " << u0 << std::endl);
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * " << u1 << std::endl);

  cv::Mat params(6,2,CV_64F);

  params.at<double>(0,0) = 0.0; //A0
  params.at<double>(1,0) = 0.0; //B0
  params.at<double>(2,0) = -1.0 * std::log((CostFunction::HIGH_VALUE - n0) / (n0 - CostFunction::LOW_VALUE)); //T0   
  params.at<double>(0,1) = u0(0); //X0
  params.at<double>(1,1) = u0(1); //Y0
  params.at<double>(2,1) = u0(2); //Z0

  params.at<double>(3,0) = 0.0; //A1
  params.at<double>(4,0) = 0.0; //B1
  params.at<double>(5,0) = -1.0 * std::log((CostFunction::HIGH_VALUE - n1) / (n1 - CostFunction::LOW_VALUE)); //T0   
  params.at<double>(3,1) = u1(0); //X1
  params.at<double>(4,1) = u1(1); //Y1
  params.at<double>(5,1) = u1(2); //Z1
  

  data.R0 = _slidingWindow.getRotation(0);
  data.R1 = _slidingWindow.getRotation(1);
  data.R2 = _slidingWindow.getRotation(2);

  std::vector<std::vector<cv::Vec3d>*> vectors{
    &(data.m0),
    &(data.m1),
    &(data.m2)
  };

  _slidingWindow.getCorrespondingFeatures(n - 1, 0, vectors);

  this->GaussNewton(data, params);

  n0 = CostFunction::scale(params.at<double>(2,0)); //T0
  n1 = CostFunction::scale(params.at<double>(5,0)); //T1
  u0 = cv::Vec3d(params.at<double>(0,1),params.at<double>(1,1),params.at<double>(2,1));
  u1 = cv::Vec3d(params.at<double>(3,1),params.at<double>(4,1),params.at<double>(5,1));

  ROS_INFO_STREAM("After: " << std::endl);
  ROS_INFO_STREAM("n0 * u0: " << n0 << " * " << u0 << std::endl);
  ROS_INFO_STREAM("n1 * u1: " << n1 << " * " << u1 << std::endl);


  st1 = (n1 * u1) + st2;
  st0 = (n0 * u0) + st1;  


  ROS_INFO_STREAM( "After: " << std::endl <<
                  "st0: " << _slidingWindow.getPosition(0) << std::endl <<
                  "st1: " << _slidingWindow.getPosition(1) << std::endl <<
                  "st2: " << _slidingWindow.getPosition(2) << std::endl);
}


double IterativeRefinement::CostFunction::func(const Input & input, const cv::Mat & params){


  auto baseLine21 = baseLine(params.at<double>(3,0), params.at<double>(4,0), params.at<double>(3,1),params.at<double>(4,1), params.at<double>(5,1));
  auto baseLine10 = baseLine(params.at<double>(0,0), params.at<double>(1,0), params.at<double>(0,1),params.at<double>(1,1), params.at<double>(2,1));
  auto scale21 = scale(params.at<double>(5,0));
  auto scale10 = scale(params.at<double>(2,0));
  auto baseLine20 = (scale21 * baseLine21 + scale10 * baseLine10) / cv::norm(scale21 * baseLine21 + scale10 * baseLine10);

  return std::pow(input.m1.dot(input.R1.t() * baseLine21.cross(input.R2 * input.m2)) ,2) +
         std::pow(input.m0.dot(input.R1.t() * baseLine10.cross(input.R1 * input.m1)) ,2) +
         std::pow(input.m0.dot(input.R0.t() * baseLine20.cross(input.R2 * input.m2)), 2);
}

cv::Vec3d IterativeRefinement::CostFunction::baseLine(double a, double b, double x, double y, double z){
  cv::Vec3d baseLine(1.0 - a * a - b * b, 2.0 * a, 2.0 * b);
  baseLine = baseLine / (1.0 + a * a + b * b);
  cv::Vec3d baseLineL;
                                                                   //TODO: faste Computation!
  baseLineL(0) = x * baseLine(0) - y * baseLine(1) - z * baseLine(2);
  baseLineL(1) = x * baseLine(1) + y * baseLine(0);
  baseLineL(2) = x * baseLine(2) + z * baseLine(0);
  return baseLineL;
}

double IterativeRefinement::CostFunction::scale(double t){
  return  LOW_VALUE + ((HIGH_VALUE - LOW_VALUE) / (1 + std::exp(-1.0 * t)));
}

double IterativeRefinement::CostFunction::derive(const Input & input, const cv::Mat & params, unsigned int index){
  auto params1 = params.clone();
  auto params2 = params.clone();

  params1.at<double>(index, 0) += DERIV_STEP;
  params2.at<double>(index, 0) -= DERIV_STEP;

  double p1 = func(input, params1);
  double p2 = func(input, params2);

  double d = (p2 - p1) / (2 * DERIV_STEP);

  return d;

}
