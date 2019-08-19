#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), _frameCounter(0), _iterativeRefinement(_slidingWindow)
{
  cv::namedWindow("t2", cv::WINDOW_NORMAL);
  cv::namedWindow("t1", cv::WINDOW_NORMAL);
  cv::namedWindow("t0", cv::WINDOW_NORMAL);
  cv::namedWindow("morph", cv::WINDOW_NORMAL);
  cv::namedWindow("rotation", cv::WINDOW_NORMAL);
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

  // Mask where Ship is in Image
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
    _cornerTracker.detectFeatures(newFeatures, image, _NUMBEROFFEATURES, std::vector<cv::Point2f>(), shipMask);
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    _slidingWindow.newFrame(std::vector<cv::Point2f>(), std::vector<cv::Vec3d>(), std::vector<uchar>(),
                            image);                                          // First Two are Dummies
    _slidingWindow.addNewFeaturesToCurrentFrame(newFeatures, newFeaturesE);  // all, because first Frame
    _slidingWindow.setPosition(b, 0);
    _slidingWindow.setRotation(R, 0);

    this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);
    //
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

    this->drawDebugPoints(trackedFeatures, cv::Scalar(0, 255, 0), _debugImage);  // TODO: use normed?
    if (!(this->checkEnoughDisparity(trackedFeatures, prevFeatures)))
    {
      OdomData od;
      od.b = b;
      od.s = _slidingWindow.getPosition(0);
      return od;
    }

    _slidingWindow.newFrame(trackedFeatures, trackedFeaturesE, found, image);

    int neededFeatures = _NUMBEROFFEATURES - _slidingWindow.getNumberOfCurrentTrackedFeatures();
    // ROS_INFO_STREAM("Current: " << _slidingWindow.getNumberOfCurrentTrackedFeatures() << std::endl);
    // ROS_INFO_STREAM("Number needed: " << neededFeatures << std::endl);
    std::vector<cv::Point2f> newFeatures(neededFeatures);
    _cornerTracker.detectFeatures(newFeatures, image, neededFeatures, trackedFeatures, shipMask);

    this->sortOutSameFeatures(trackedFeatures, newFeatures);  // TODO: maybe not needed any more
    std::vector<cv::Vec3d> newFeaturesE;
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    _slidingWindow.addNewFeaturesToCurrentFrame(newFeatures, newFeaturesE);

    this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);

    cv::Matx33d rBefore = _slidingWindow.getRotation(1);
    cv::Matx33d rDiff = rBefore.t() * R;  // Difference Rotation
    //
    std::vector<cv::Vec3d> thisCorespFeaturesE, beforeCorespFeaturesE;
    _slidingWindow.getCorrespondingFeatures(1, 0, beforeCorespFeaturesE, thisCorespFeaturesE);
    std::vector<cv::Vec3d> thisCorespFeaturesEUnrotate;
    this->unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesEUnrotate, rDiff);

    //
    std::vector<unsigned int> inlier;
    std::vector<cv::Point2f> outLierDraws;
    b = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesEUnrotate, inlier);
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

      /*------> * Debug Features -> History **/
      std::vector<cv::Point2f> ft0, ft1, ft2;
      std::vector<std::vector<cv::Point2f> *> vectors{ &ft0, &ft1, &ft2 };

      _slidingWindow.getCorrespondingFeatures(2, 0, vectors);
      cv::Mat t2 = _slidingWindow.getImage(2).clone();
      cv::Mat t1 = _slidingWindow.getImage(1).clone();
      cv::Mat t0 = _slidingWindow.getImage(0).clone();
      cv::Mat morph(t0.size(), CV_8UC3, cv::Scalar(0));
      cv::Mat rotation(t0.size(), CV_8UC3, cv::Scalar(0));
      for (unsigned int i = 0; i < thisCorespFeaturesEUnrotate.size(); i++)
      {
        cv::Point2d pt = cameraModel.project3dToPixel(thisCorespFeaturesE[i]); //ROT
        cv::Point2d ptU = cameraModel.project3dToPixel(thisCorespFeaturesEUnrotate[i]); //Grün
        cv::Point2d ptBU = cameraModel.project3dToPixel(beforeCorespFeaturesE[i]); //Blau
        cv::circle(rotation, pt, 5, cv::Scalar(0, 0, 255), -1);
        cv::line(rotation, pt, ptU, cv::Scalar(255, 255, 255), 4);
        cv::circle(rotation, ptU, 5, cv::Scalar(0, 255, 0), -1);
        cv::line(rotation, pt, ptBU, cv::Scalar(255, 255, 255), 4);
        cv::circle(rotation, ptBU, 5, cv::Scalar(255, 0, 0), -1);
      }

      assert(ft0.size() == ft1.size() && ft1.size() == ft2.size());

      for (auto out : outLierDraws)
      {
        cv::circle(morph, out, 10, cv::Scalar(0, 255, 255), -1);
      }

      for (unsigned int i = 0; i < ft0.size(); i++)
      {
        cv::circle(morph, ft2[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(morph, ft2[i], ft1[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morph, ft1[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::line(morph, ft1[i], ft0[i], cv::Scalar(255, 255, 255), 4);
        cv::circle(morph, ft0[i], 5, cv::Scalar(255, 0, 0), -1);
      }

      this->drawDebugPoints(ft2, cv::Scalar(0, 0, 255), t2);
      this->drawDebugPoints(ft1, cv::Scalar(0, 0, 255), t1);
      this->drawDebugPoints(ft0, cv::Scalar(0, 0, 255), t0);

      cv::imshow("t2", t2);
      cv::imshow("t1", t1);
      cv::imshow("t0", t0);

      cv::imshow("morph", morph);
      cv::imshow("rotation", rotation);

      cv::waitKey(1);
      /*<----- END Debug */

      cv::Vec3d st = _slidingWindow.getPosition(1) + b;
      _slidingWindow.setPosition(st, 0);
      _slidingWindow.setRotation(R, 0);
      _iterativeRefinement.refine(3);

      this->drawDebugImage(_slidingWindow.getPosition(0) - _slidingWindow.getPosition(1), _debugImage,
                           cv::Scalar(0, 0, 255));
    }
    else
    {
      _slidingWindow.setPosition(_slidingWindow.getPosition(1) + sign * b, 0);
      _slidingWindow.setRotation(R, 0);
    }
    this->drawDebugImage(sign * b, _debugImage, cv::Scalar(0, 255, 0));
  }

  _frameCounter++;
  OdomData od;
  od.b = b;
  od.s = _slidingWindow.getPosition(0);
  return od;
}

void MVO::sortOutSameFeatures(const std::vector<cv::Point2f> &beforeFeatures, std::vector<cv::Point2f> &newFeatures)
{
  for (auto beforeFeature = beforeFeatures.begin(); beforeFeature != beforeFeatures.end(); beforeFeature++)
  {
    for (auto newFeature = newFeatures.begin(); newFeature != newFeatures.end(); newFeature++)
    {
      double distance = cv::norm((*beforeFeature) - (*newFeature));
      if (distance < 20)
      {  // TODO: Pram Threshold
        newFeatures.erase(newFeature);
        break;
      }
    }
  }
}

void MVO::euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> &featuresE,
                             const image_geometry::PinholeCameraModel &cameraModel)
{
  for (auto feature = features.begin(); feature != features.end(); feature++)
  {
    featuresE.push_back(cameraModel.projectPixelTo3dRay(*feature));
  }
}

void MVO::drawDebugImage(const cv::Vec3d baseLine, cv::Mat &image, const cv::Scalar &color)
{
  auto baseLineNorm = cv::normalize(baseLine);

  int mitX = double(image.cols) / 2.0;
  int mitY = double(image.rows) / 2.0;
  double scaleX = (image.cols - mitX) / 1.5;
  double scaleY = (image.rows - mitY) / 1.5;
  cv::arrowedLine(image, cv::Point(mitX, mitY),
                  cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY), color, 10);
  cv::line(image, cv::Point(mitY, 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), 20), color, 10);
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

bool MVO::checkEnoughDisparity(const std::vector<cv::Point2f> &first, const std::vector<cv::Point2f> &second)
{
  assert(first.size() == second.size());
  //
  double diff = 0;
  for (auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++)
  {
    diff += cv::norm((*p1) - (*p2));
    //
  }
  diff = diff / first.size();
  //
  return diff > 20;  // TODO: Thresh
}

void MVO::unrotateFeatures(const std::vector<cv::Vec3d> &features, std::vector<cv::Vec3d> &unrotatedFeatures,
                           const cv::Matx33d &R)
{
  ROS_INFO_STREAM("Rotation: " << R << std::endl);
  for (auto feature = features.begin(); feature != features.end(); feature++)
  {
    auto unrotatedFeature = R * (*feature);
    unrotatedFeature = unrotatedFeature / unrotatedFeature(2);
    unrotatedFeatures.push_back(unrotatedFeature);
  }
}
