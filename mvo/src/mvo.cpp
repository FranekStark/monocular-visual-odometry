#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), _frameCounter(0), _iterativeRefinement(_slidingWindow)
{
  cv::namedWindow("t2", cv::WINDOW_NORMAL);
  cv::namedWindow("t1", cv::WINDOW_NORMAL);
  cv::namedWindow("t0", cv::WINDOW_NORMAL);
  cv::namedWindow("morphBef", cv::WINDOW_NORMAL);
  cv::namedWindow("morphAfter", cv::WINDOW_NORMAL);
  cv::namedWindow("rotated", cv::WINDOW_NORMAL);
}

MVO::~MVO()
{
  cv::destroyAllWindows();
}

OdomData MVO::handleImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel,
                          const cv::Matx33d &R)
{
  cv::cvtColor(image, _debugImage, cv::ColorConversionCodes::COLOR_GRAY2RGB);
  cv::Vec3d b(0, 0, 0);

  /**
   *  Mask, where Ship is In Image
   **/
  cv::Rect2d shipMask((image.size().width / 2) - (1.6 / 16.0) * image.size().width,
                      image.size().height - (6.5 / 16.0) * image.size().height, (3.2 / 16.0) * image.size().width,
                      (6.5 / 16.0) * image.size().width);
  cv::Mat maskImage = _debugImage.clone();
  cv::rectangle(maskImage, shipMask, cv::Scalar(0, 0, 255), -1);
  cv::addWeighted(_debugImage, 0.5, maskImage, 0.5, 0.0, _debugImage);

  if (_frameCounter == 0)
  {
    //
    std::vector<cv::Point2f> newFeatures;
    std::vector<cv::Vec3d> newFeaturesE;
    _cornerTracker.detectFeatures(newFeatures, image, _NUMBEROFFEATURES, std::vector<cv::Point2f>(), shipMask, true);
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    _slidingWindow.newFrame(std::vector<cv::Point2f>(), std::vector<cv::Vec3d>(), std::vector<uchar>(),
                            image);                                          // First Two are Dummies
    _slidingWindow.addNewFeaturesToCurrentFrame(newFeatures, newFeaturesE);  // all, because first Frame
    _slidingWindow.setPosition(b, 0);
    _slidingWindow.setRotation(R, 0);

    this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);
    //
    _slidingWindow.persistCurrentFrame();
  }
  else
  {
    std::vector<cv::Point2f> prevFeatures;
    _slidingWindow.getFeatures(0, prevFeatures);  // Currently Frame 0 is the "previous"
    cv::Mat prevImage = _slidingWindow.getImage(0);
    std::vector<cv::Point2f> trackedFeatures;
    std::vector<cv::Vec3d> trackedFeaturesE;
    std::vector<unsigned char> found;
    //
    _cornerTracker.trackFeatures(prevImage, image, prevFeatures, trackedFeatures, found, shipMask);
    //
    this->euclidNormFeatures(trackedFeatures, trackedFeaturesE, cameraModel);
    _slidingWindow.newFrame(trackedFeatures, trackedFeaturesE, found, image);
    int neededFeatures = _NUMBEROFFEATURES - _slidingWindow.getNumberOfCurrentTrackedFeatures();
    std::vector<cv::Point2f> newFeatures(neededFeatures);
    _cornerTracker.detectFeatures(newFeatures, image, neededFeatures, trackedFeatures, shipMask, false);
    std::vector<cv::Vec3d> newFeaturesE;
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    _slidingWindow.addNewFeaturesToCurrentFrame(newFeatures, newFeaturesE);
    this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);
    this->drawDebugPoints(trackedFeatures, cv::Scalar(0, 255, 0), _debugImage);  // TODO: use normed?
    
    /**
     * Check  for enaough Disparity
     **/
    std::vector<cv::Vec3d> thisCorespFeaturesE, beforeCorespFeaturesE;
    _slidingWindow.getCorrespondingFeatures(1, 0, beforeCorespFeaturesE, thisCorespFeaturesE);


    /**
     * Debug Images For Rotation View
     **/
    {
      auto imageRot = cv::Mat(image.size(), CV_8SC3, cv::Scalar(0));
      std::vector<cv::Vec3d> beforeCorrespFeaturesE, nowCorrespFeaturesE, nowCorrespFeaturesUnrotatedE;
      _slidingWindow.getCorrespondingFeatures(1,0,beforeCorrespFeaturesE, nowCorrespFeaturesE);

      auto beforeRot = _slidingWindow.getRotation(1);



      auto diffRot = beforeRot.t() * R;
      this->unrotateFeatures(nowCorrespFeaturesE, nowCorrespFeaturesUnrotatedE, diffRot);

      std::vector<cv::Point2f> beforeCorrespFeatures, nowCorrespFeaturesUnrotated, nowCorrespFeatures;
      auto beforeCorrespFeaturesEIt = beforeCorrespFeaturesE.begin();
      auto nowCorrespFeaturesUnrotatedEIt = nowCorrespFeaturesUnrotatedE.begin();
      auto nowCorrespFeaturesEIt = nowCorrespFeaturesE.begin();
      while(beforeCorrespFeaturesEIt != beforeCorrespFeaturesE.end()){
        beforeCorrespFeatures.push_back(cameraModel.project3dToPixel(*beforeCorrespFeaturesEIt));
        nowCorrespFeaturesUnrotated.push_back(cameraModel.project3dToPixel(*nowCorrespFeaturesUnrotatedEIt));
        nowCorrespFeatures.push_back(cameraModel.project3dToPixel(*nowCorrespFeaturesEIt));
        beforeCorrespFeaturesEIt++;
        nowCorrespFeaturesUnrotatedEIt++;
        nowCorrespFeaturesEIt++;
      }
      this->drawDebugPoints(beforeCorrespFeatures, cv::Scalar(0,0,255), imageRot);
      this->drawDebugPoints(nowCorrespFeaturesUnrotated, cv::Scalar(255,0,0), imageRot);
      this->drawDebugPoints(nowCorrespFeatures, cv::Scalar(0,255,0), imageRot);

      _debugImage2 = imageRot;
      //cv::imshow("rotated", imageRot);
      cv::waitKey(1);

      //ROS_INFO_STREAM("RotationMatrix: " << diffRot << std::endl);
     
      float sy = sqrt(diffRot(0,0) * diffRot(0,0) +  diffRot(1,0) * diffRot(1,0) );
  
      bool singular = sy < 1e-6; // If
  
      float x, y, z;
      if (!singular)
      {
          x = atan2(diffRot(2,1) , diffRot(2,2));
          y = atan2(-diffRot(2,0), sy);
          z = atan2(diffRot(1,0), diffRot(0,0));
      }
      else
      {
          x = atan2(-diffRot(1,2), diffRot(1,1));
          y = atan2(-diffRot(2,0), sy);
          z = 0;
      }
      ROS_INFO_STREAM("Decomposed: " << "yaw: " << z*(180.0/3.141592653589793238463) <<", pitch: " << y*(180.0/3.141592653589793238463) <<", roll: " << x*(180.0/3.141592653589793238463) << std::endl);


      }

    if (true || !(this->checkEnoughDisparity(beforeCorespFeaturesE, thisCorespFeaturesE)))
    {
      OdomData od;
      od.b = b;
      od.s = _slidingWindow.getPosition(0);
      return od;
    }

    //Has enaugh Disparity
    _slidingWindow.persistCurrentFrame();
    cv::Matx33d rBefore = _slidingWindow.getRotation(1);
    cv::Matx33d rDiff = rBefore.t() * R;  // Difference Rotation
   

    //

    std::vector<cv::Vec3d> thisCorespFeaturesEUnrotate;
    this->unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesEUnrotate, rDiff);

    //
    std::vector<unsigned int> inlier;
    std::vector<cv::Point2f> outLierDraws;
    std::vector<Feature> outLier;
    b = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesEUnrotate, inlier);
    cv::Mat morphBeforeRemove(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat morphAfterRemove(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    if (_frameCounter > 1)
    {
      std::vector<cv::Point2f> ft0, ft1, ft2;
      std::vector<std::vector<cv::Point2f> *> vectors{ &ft0, &ft1, &ft2 };
      _slidingWindow.getCorrespondingFeatures(2, 0, vectors);

      for (unsigned int i = 0; i < ft0.size(); i++)
      {
        cv::circle(morphBeforeRemove, ft2[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(morphBeforeRemove, ft2[i], ft1[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morphBeforeRemove, ft1[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::line(morphBeforeRemove, ft1[i], ft0[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morphBeforeRemove, ft0[i], 5, cv::Scalar(255, 0, 0), -1);
      }
    }

    // Remove Outlier //TODO: very expensive!
    for (auto feature = thisCorespFeaturesE.begin(); feature != thisCorespFeaturesE.end(); feature++)
    {
      unsigned int index = std::distance(thisCorespFeaturesE.begin(), feature);
      bool found = false;
      for (auto inlierIndex : inlier)
      {
        if (inlierIndex == index)
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        _slidingWindow.removeFeatureFromCurrentWindow(*feature);
        outLierDraws.push_back(cv::Point2f(cameraModel.project3dToPixel(*feature)));
      }
    }

    for (auto out : outLierDraws)
    {
      cv::circle(morphBeforeRemove, out, 10, cv::Scalar(0, 255, 255), -1);
      cv::circle(morphAfterRemove, out, 10, cv::Scalar(0, 255, 255), -1);
    }

    if (_frameCounter > 1)
    {
      std::vector<cv::Point2f> ft0, ft1, ft2;
      std::vector<std::vector<cv::Point2f> *> vectors{ &ft0, &ft1, &ft2 };
      _slidingWindow.getCorrespondingFeatures(2, 0, vectors);

      for (unsigned int i = 0; i < ft0.size(); i++)
      {
        cv::circle(morphAfterRemove, ft2[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(morphAfterRemove, ft2[i], ft1[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morphAfterRemove, ft1[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::line(morphAfterRemove, ft1[i], ft0[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morphAfterRemove, ft0[i], 5, cv::Scalar(255, 0, 0), -1);
      }
    }
    cv::imshow("morphBef", morphBeforeRemove);
    cv::imshow("morphAfter", morphAfterRemove);
    cv::waitKey(1);
    _debugImage2 = morphAfterRemove;

    //
    // Scale vote
    std::vector<double> depths(beforeCorespFeaturesE.size());
    //
    this->reconstructDepth(depths, thisCorespFeaturesE, beforeCorespFeaturesE, rDiff, b);
    double sign = 0;
    for (auto depth = depths.begin(); depth != depths.end(); depth++)
    {
      if ((*depth) < 0)
      {
        sign--;
      }
      else if ((*depth) > 0)
      {
        sign++;
      }
    }

    if (sign < 0)
    {
      sign = -1;
      b = -1.0 * b;
    }
    else
    {
      sign = 1;
    }
    sign = 1.0;

    if (_frameCounter > 1)
    {  // IterativeRefinemen -> Scale Estimation?

      cv::Vec3d st = _slidingWindow.getPosition(1) + b;
      _slidingWindow.setPosition(st, 0);
      _slidingWindow.setRotation(R, 0);
      // cv::waitKey(0);
      _slidingWindow.exportMatlabData();
      // cv::waitKey(0);
      _iterativeRefinement.refine(3);

      this->drawDebugImage(_slidingWindow.getPosition(0) - _slidingWindow.getPosition(1), _debugImage2,
                           cv::Scalar(0, 0, 255), 1);
    }
    else
    {
      _slidingWindow.setPosition(_slidingWindow.getPosition(1) + sign * b, 0);
      _slidingWindow.setRotation(R, 0);
    }
    this->drawDebugImage(sign * b, _debugImage2, cv::Scalar(0, 255, 0), 2);
    this->drawDebugScale(_debugImage2, 1, cv::norm(_slidingWindow.getPosition(0)));
  }

  _frameCounter++;
  OdomData od;
  od.b = b;
  od.s = _slidingWindow.getPosition(0);

  return od;
}


void MVO::euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> &featuresE,
                             const image_geometry::PinholeCameraModel &cameraModel)
{
  for (auto feature = features.begin(); feature != features.end(); feature++)
  {
    featuresE.push_back(cameraModel.projectPixelTo3dRay(*feature));
  }
}

void MVO::drawDebugImage(const cv::Vec3d &baseLine, cv::Mat &image, const cv::Scalar &color, unsigned int index)
{
  auto baseLineNorm = cv::normalize(baseLine);

  int mitX = double(image.cols) / 2.0;
  int mitY = double(image.rows) / 2.0;
  double scaleX = (image.cols - mitX) / 1.5;
  double scaleY = (image.rows - mitY) / 1.5;
  cv::arrowedLine(image, cv::Point(mitX, mitY),
                  cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY), color, 10);
  cv::line(image, cv::Point(mitY, index * 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), index * 20), color, 10);
}

void MVO::drawDebugScale(cv::Mat image, double scaleBefore, double scaleAfter)
{
  cv::rectangle(image, cv::Rect(10, 10, 40, image.rows - 20), cv::Scalar(0, 0, 255), 4);
  double scaling = scaleAfter / scaleBefore;
  cv::rectangle(image, cv::Rect(12, 10, 8, scaling * (image.rows - 20)), cv::Scalar(0, 255, 0), -1);
}

void MVO::drawDebugPoints(const std::vector<cv::Point2f> &points, const cv::Scalar &color, cv::Mat &image)
{
  for (auto point = points.begin(); point != points.end(); point++)
  {
    cv::circle(image, cv::Point(*point), 10, color, -10);
    std::string index;
    cv::putText(image, std::to_string(std::distance(points.begin(), point)), cv::Point(*point), cv::FONT_HERSHEY_PLAIN,
                0.5, cv::Scalar(255, 255, 255));
  }
}

void MVO::reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                           const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r, const cv::Vec3d &b)
{
  assert(m1L.size() == m2L.size());
  for (auto m1 = m1L.begin(), m2 = m2L.begin(); m1 != m1L.end() && m2 != m2L.end(); m1++, m2++)
  {
    cv::Matx33d C;
    C << 1, 0, -(*m2)(0),  //
        0, 1, -(*m2)(1),   //
        -(*m2)(0), -(*m2)(1), (*m2)(0) * (*m2)(0) + (*m2)(1) * (*m2)(1);
    double Z = (m1->t() * r * C * r.t() * b)(0) / (m1->t() * r * C * r.t() * (*m1))(0);
    depth.push_back(Z);
  }
}

bool MVO::checkEnoughDisparity(const std::vector<cv::Vec3d> &first, const std::vector<cv::Vec3d> &second)
{
  assert(first.size() == second.size());
  //
  double diff = 0;
  for (auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++)
  {
    diff += cv::norm((p2) - (p1));
  }
  diff = diff / first.size();
  //
  return diff > (2.8284 * 0.10);  //--> 10% Abweicung// TODO: Thresh --> Zwischen sqrt(2²+2²) und 0
}

void MVO::unrotateFeatures(const std::vector<cv::Vec3d> &features, std::vector<cv::Vec3d> &unrotatedFeatures,
                           const cv::Matx33d &R)
{
  for (auto feature = features.begin(); feature != features.end(); feature++)
  {
    auto unrotatedFeature = R * (*feature);
    unrotatedFeature = unrotatedFeature / unrotatedFeature(2);
    unrotatedFeatures.push_back(unrotatedFeature);
  }
}
