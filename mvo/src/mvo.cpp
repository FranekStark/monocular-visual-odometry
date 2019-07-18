#include "mvo.hpp"

#include <map>

MVO::MVO()
{
  cv::namedWindow("original", cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow("grayScale", cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow("cornerImage", cv::WINDOW_GUI_EXPANDED);
  cv::startWindowThread();
}

MVO::~MVO()
{
  cv::destroyWindow("original");
  cv::destroyWindow("cornerImage");
}

void MVO::handleImage(const cv::Mat &image)
{
  /*Original */
  cv::imshow("original", image);
  cv::waitKey(10);
  /*Grayscale */
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY, 3);
  /*Corners */
  std::list<cv::Point> corners = this->detectCorners(grayImage, 20);
  ROS_INFO_STREAM("Anzahl Corner: " << corners.size() << std::endl);

  if (corners.size() < 2300000)  // TODO: quickfix, weil er sonst ewig rechnet
  {
    cv::Mat cornerImage = image.clone();

    /*Mark Corners*/
    for (auto const &corner : corners)
    {
      cv::circle(cornerImage, corner, 10, cv::Scalar(0, 0, 255), -10);
    }
    imshow("cornerImage", cornerImage);
  }
}

// Must be Grayscale
std::list<cv::Point> MVO::detectCorners(const cv::Mat &image, int num)
{
  int blockSize = 8;  // TODO:Params
  int apertureSize = 3;
  double k = 0.04;
  int thresh = 150;

  std::map<int, std::list<cv::Point>> corners;

  cv::Mat cornerImage = cv::Mat::zeros(image.size(), CV_32FC1);      // DST vorbereiten
  cv::cornerHarris(image, cornerImage, blockSize, apertureSize, k);  // Corners berechnen
  cv::Mat cornerImageNorm;
  cv::normalize(cornerImage, cornerImageNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1,
                cv::Mat());  // auf 0...255 normalisieren

  // Die num besten finden:
  for (int i = 0; i < cornerImageNorm.rows; i++)
  {
    for (int j = 0; j < cornerImageNorm.cols; j++)
    {
      int weight = (int)cornerImageNorm.at<float>(i, j);
      if (weight > thresh)
      {
        corners[weight].push_back(cv::Point(j, i));
      }
    }
  }

  std::list<cv::Point> result;
  int cornersCnt = 0;

  for (std::map<int, std::list<cv::Point>>::reverse_iterator rit = corners.rbegin(); rit != corners.rend(); rit++)
  {
    for (cv::Point point : rit->second)
    {
      if (cornersCnt >= num)
      {
        return result;
      }
      result.push_back(point);
      cornersCnt++;
    }
  }
  return result;
}