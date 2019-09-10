#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), 
_frameCounter(0),
_numberOfFeatures(60), 
_disparityThreshold(0.01),
_iterativeRefinement(_slidingWindow)
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
  ROS_INFO_STREAM("hanlde Image" << std::endl);
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

  /**
   * When FrameCounter == 0, this is the First Image
   **/
  if (_frameCounter == 0)
  {
    //
    std::vector<cv::Point2f> newFeatures;
    std::vector<cv::Vec3d> newFeaturesE;
    _cornerTracker.detectFeatures(newFeatures, image, _numberOfFeatures, std::vector<cv::Point2f>(), shipMask, false);
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    _slidingWindow.newFrame(std::vector<cv::Point2f>(), std::vector<cv::Vec3d>(), std::vector<uchar>(),
                            image);                                          // First Two are Dummies
    _slidingWindow.addNewFeaturesToFrame(newFeatures, newFeaturesE,0);  // all, because first Frame
    this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);
    _slidingWindow.setPosition(b, 0);
    _slidingWindow.setRotation(R,0);
    _slidingWindow.persistCurrentFrame();
  }
  else
  {

    /**
     * Calculate Rotation between Images
     **/
    auto beforeImageRot = _slidingWindow.getRotation(0); //CurentImage is before Image
    auto diffImageRot = beforeImageRot * R;


    /**
     * Get Previous Features and Track them
     **/
    std::vector<cv::Point2f> prevFeatures;
    _slidingWindow.getFeatures(0, prevFeatures);  // Currently Frame 0 is the "previous"
    cv::Mat prevImage = _slidingWindow.getImage(0);
    std::vector<cv::Point2f> trackedFeatures;
    std::vector<cv::Vec3d> trackedFeaturesE, trackedFeaturesEUnrotaed;
    std::vector<unsigned char> found;
    this->euclidNormFeatures(trackedFeatures, trackedFeaturesE, cameraModel);
    this->unrotateFeatures(trackedFeaturesE, trackedFeaturesEUnrotaed,diffImageRot.t());
    this->euclidUnNormFeatures(trackedFeaturesEUnrotaed, trackedFeatures, cameraModel);
    _cornerTracker.trackFeatures(prevImage, image, prevFeatures, trackedFeatures, found, shipMask);
    trackedFeaturesE.clear();
    this->euclidNormFeatures(trackedFeatures, trackedFeaturesE, cameraModel);

    /**
     * This is a new Image on a Slidly different Position
     **/
    
    _slidingWindow.newFrame(trackedFeatures, trackedFeaturesE, found, image);
    _slidingWindow.setRotation(R,0); //Set Current Rotation

    /**
     * Get the Corrsponding Featurs of this and the Frame before
     **/
    std::vector<cv::Vec3d> thisCorespFeaturesE, beforeCorespFeaturesE;
    _slidingWindow.getCorrespondingFeatures(1, 0, beforeCorespFeaturesE, thisCorespFeaturesE);
    /**
     * Calculate Rotation between Frames
     **/
    std::vector<cv::Vec3d> thisCorespFeaturesUnrotatedE;
    auto beforeFrameRot = _slidingWindow.getRotation(1); //CurentFrame is before Frame
    auto diffFrameRot = beforeFrameRot.t() * R;
    /**
     * Unrotate Features
     **/
    this->unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesUnrotatedE, diffFrameRot);
    /**
     * Calc Disparity
     */
    auto diff = this->calcDisparity(thisCorespFeaturesUnrotatedE, beforeCorespFeaturesE);
    ROS_INFO_STREAM("Diff: " << diff << "/" << 2.8284 * _disparityThreshold << std::endl);
    /**
     * Detect new Features, if required
     **/
    int neededFeatures = _numberOfFeatures - _slidingWindow.getNumberOfCurrentTrackedFeatures();
    ROS_INFO_STREAM("Needed Features: " << neededFeatures << std::endl);
    std::vector<cv::Point2f> newFeatures(neededFeatures);
    _cornerTracker.detectFeatures(newFeatures, image, neededFeatures, trackedFeatures, shipMask, false);
    std::vector<cv::Vec3d> newFeaturesE;
    this->euclidNormFeatures(newFeatures, newFeaturesE, cameraModel);
    if(diff < 0.001) //SamePosition, as Before so add New Features to FrameBefore
    {
      _slidingWindow.addNewFeaturesToBeforeFrame(newFeatures, newFeaturesE);
      beforeCorespFeaturesE.insert(beforeCorespFeaturesE.end(), newFeaturesE.begin(), newFeaturesE.end());
      thisCorespFeaturesE.insert(thisCorespFeaturesE.end(), newFeaturesE.begin(), newFeaturesE.end());
      std::vector<cv::Vec3d> newFeaturesEUnrotated;
      this->unrotateFeatures(newFeaturesE, newFeaturesEUnrotated, diffFrameRot);
      thisCorespFeaturesUnrotatedE.insert(thisCorespFeaturesUnrotatedE.end(), newFeaturesEUnrotated.begin(), newFeaturesEUnrotated.end());
    }else{
      _slidingWindow.addNewFeaturesToFrame(newFeatures, newFeaturesE,0);
    }
    #ifdef DEBUGIMAGES
      this->drawDebugPoints(newFeatures, cv::Scalar(0, 0, 255), _debugImage);
      this->drawDebugPoints(trackedFeatures, cv::Scalar(0, 255, 0), _debugImage);  // TODO: use normed?
    #endif




    #ifdef DEBUGIMAGES
      
      {
        _debugImage3 = cv::Mat(image.size(), CV_8SC3, cv::Scalar(0));
        std::vector<cv::Point2f> ft0, ft1, ft2;
        std::vector<std::vector<cv::Point2f> *> vectors{ &ft0, &ft1};
        if (_frameCounter > 1){
          vectors.push_back(&ft2 );
         _slidingWindow.getCorrespondingFeatures(2, 0, vectors);
        }else{
         _slidingWindow.getCorrespondingFeatures(1, 0, vectors);
        }

        for (unsigned int i = 0; i < ft0.size(); i++)
        {
          if (_frameCounter > 1){
          cv::circle(_debugImage3, ft2[i], 5, cv::Scalar(0, 0, 255), -1);
          cv::line(_debugImage3, ft2[i], ft1[i], cv::Scalar(255, 255, 255), 4);
          }
          cv::circle(_debugImage3, ft1[i], 5, cv::Scalar(0, 255, 0), -1);
          cv::line(_debugImage3, ft1[i], ft0[i], cv::Scalar(255, 255, 255), 4);
          cv::circle(_debugImage3, ft0[i], 5, cv::Scalar(255, 0, 0), -1);
        }
      }
    #endif


    /**
     * Debug Images For Rotation View
     **/
    #ifdef DEBUGIMAGES
    {
      _debugImage4 = cv::Mat(image.size(), CV_8SC3, cv::Scalar(0));
      std::vector<cv::Point2f> beforeCorrespFeatures, nowCorrespFeaturesUnrotated, nowCorrespFeatures;
      auto beforeCorrespFeaturesEIt = beforeCorespFeaturesE.begin();
      auto nowCorrespFeaturesUnrotatedEIt = thisCorespFeaturesUnrotatedE.begin();
      auto nowCorrespFeaturesEIt = thisCorespFeaturesE.begin();
      while(beforeCorrespFeaturesEIt != beforeCorespFeaturesE.end()){
        beforeCorrespFeatures.push_back(cameraModel.project3dToPixel(*beforeCorrespFeaturesEIt));
        nowCorrespFeaturesUnrotated.push_back(cameraModel.project3dToPixel(*nowCorrespFeaturesUnrotatedEIt));
        nowCorrespFeatures.push_back(cameraModel.project3dToPixel(*nowCorrespFeaturesEIt));
        beforeCorrespFeaturesEIt++;
        nowCorrespFeaturesUnrotatedEIt++;
        nowCorrespFeaturesEIt++;
      }
      this->drawDebugPoints(beforeCorrespFeatures, cv::Scalar(0,0,255), _debugImage4);
      this->drawDebugPoints(nowCorrespFeaturesUnrotated, cv::Scalar(255,0,0), _debugImage4);
      this->drawDebugPoints(nowCorrespFeatures, cv::Scalar(0,255,0), _debugImage4);
   }
   #endif
    

    
    /**
     * Check  for enaough Disparity and Enough Points
     **/
    if ( diff < (2.8284 * _disparityThreshold))  //--> 10% Abweicung// TODO: Thresh --> Zwischen sqrt(2²+2²) und 0)
    {
      OdomData od;
      od.b = b;
      od.s = _slidingWindow.getPosition(1);
      od.o = R;
      return od; //In Case if there is not enough Disparity between the Points, the Function returns with same Position.
    }else if(thisCorespFeaturesUnrotatedE.size() <= 4){ //TODO: How many?
      ROS_WARN_STREAM("Lost almost all tracking Points, scaling lost..." << std::endl);
      _slidingWindow.setPosition(_slidingWindow.getPosition(1),0);
      _slidingWindow.setRotation(R,0);
      _slidingWindow.persistCurrentFrame();
      OdomData od;
      od.b = b;
      od.s = _slidingWindow.getPosition(0);
      od.o = _slidingWindow.getRotation(0);
      _frameCounter++;
      return od; //In Case there aren't enough Features anynore, persist this Frame and return. Because this is the new Last Frame, uppon we estimate new Motions. 
    }
   

        

    /**
     * The Points have enough Disparity, so we will persist this Frame and start a new Algorithm
     **/
    _slidingWindow.persistCurrentFrame();
    cv::Matx33d rBefore = _slidingWindow.getRotation(1);
    cv::Matx33d rDiff = rBefore.t() * R;  // Difference Rotation
   

    /**
     * First Guess of the Direction-BaseLine
     */
    std::vector<unsigned int> inlier;
    std::vector<cv::Point2f> outLierDraws;
    std::vector<Feature> outLier;
    b = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesUnrotatedE, inlier);
    cv::Mat morphBeforeRemove(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat morphAfterRemove(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    #ifdef DEBUGIMAGES
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
    #endif

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
        #ifdef DEBUGIMAGES
          outLierDraws.push_back(cv::Point2f(cameraModel.project3dToPixel(*feature)));
        #endif
      }
    }

    #ifdef DEBUGIMAGES
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
    #endif

    /**
     * Vote for the sign of the Baseline, which generates the feweset negative Gradients
     **/
    std::vector<double> depths(beforeCorespFeaturesE.size());
    std::vector<double> depthsNegate(beforeCorespFeaturesE.size());
    auto bnegate = -1.0 * b;
    this->reconstructDepth(depths, thisCorespFeaturesE, beforeCorespFeaturesE, rDiff, b);
    this->reconstructDepth(depthsNegate, thisCorespFeaturesE, beforeCorespFeaturesE, rDiff, bnegate);
    unsigned int negCountb = 0;
    unsigned int negCountbnegate = 0;
    assert(depths.size() == depthsNegate.size());
    auto depthIt = depths.begin();
    auto depthnegateIt = depthsNegate.begin();
    while(depthIt != depths.end()){
      if(*depthIt < 0){ //TODO: Count here also for vanishing depths?
        negCountb++;
      }
      if(*depthnegateIt < 0){
        negCountbnegate++;
      }
      depthIt++;
      depthnegateIt++;
    }
    if(negCountb < negCountbnegate){
      b = b;
    }else if(negCountbnegate < negCountb){
      b = bnegate;
    }
    else{
      ROS_WARN_STREAM("Couldn't find unambiguous solution for sign of movement." << std::endl);
    } 


    /**
     * Transform the relative BaseLine into WorldCordinates
     */
    b = beforeFrameRot * b;

    if (_frameCounter > 1)
    {  // IterativeRefinemen -> Scale Estimation?

      #ifdef DEBUGIMAGES
        cv::Vec3d b1Before = _slidingWindow.getRotation(1).t() * (_slidingWindow.getPosition(2) - _slidingWindow.getPosition(1));
      #endif

      cv::Vec3d st = _slidingWindow.getPosition(1) + b;
      _slidingWindow.setPosition(st, 0);
      _slidingWindow.setRotation(R, 0);
      // cv::waitKey(0);
      _slidingWindow.exportMatlabData();
      // cv::waitKey(0);
      _iterativeRefinement.refine(3);
      #ifdef DEBUGIMAGES
        this->drawDebugImage(_slidingWindow.getRotation(0).t()*(_slidingWindow.getPosition(0) - _slidingWindow.getPosition(1)), _debugImage2,
                            cv::Scalar(0, 0, 255), 1);

        /**
         * Small Image with before Motion
         */
        cv::Vec3d b1After = _slidingWindow.getRotation(1).t()*(_slidingWindow.getPosition(2) - _slidingWindow.getPosition(1));
        cv::Rect subImageField(_debugImage2.cols / 2, _debugImage2.rows / 2, (_debugImage2.cols / 2) - 1, (_debugImage2.rows / 2) - 1); //SubImage unteres rechtes Viertel
        cv::Mat smallDebug2Image = _debugImage2(subImageField);
        this->drawDebugImage(b1Before, smallDebug2Image, cv::Scalar(0,255,0), 0);
        this->drawDebugImage(b1After, smallDebug2Image, cv::Scalar(0,0,255), 1);
      #endif
    }
    else
    {
      _slidingWindow.setPosition(_slidingWindow.getPosition(1) + b, 0);
      _slidingWindow.setRotation(R, 0);
    }

    #ifdef DEBUGIMAGES
      this->drawDebugImage(_slidingWindow.getRotation(0).t() * b, _debugImage2, cv::Scalar(0, 255, 0), 2);
      this->drawDebugScale(_debugImage2, 1, cv::norm(_slidingWindow.getPosition(0)));
    #endif
  }

  _frameCounter++;
  OdomData od;
  od.b = b;
  od.s = _slidingWindow.getPosition(0);
  od.o = _slidingWindow.getRotation(0);

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

void MVO::euclidUnNormFeatures(const std::vector<cv::Vec3d> &featuresE, std::vector<cv::Point2f> & features, const image_geometry::PinholeCameraModel & cameraModel){
  for(auto featureE = featuresE.begin(); featureE != featuresE.end(); featureE++){
    features.push_back(cameraModel.project3dToPixel(*featureE));
  }
}


void MVO::drawDebugImage(const cv::Vec3d &baseLine, cv::Mat &image, const cv::Scalar &color, unsigned int index)
{
  double norm = cv::norm(baseLine);
  auto baseLineNorm = cv::normalize(baseLine);

  int mitX = double(image.cols) / 2.0;
  int mitY = double(image.rows) / 2.0;
  double scaleX = (image.cols - mitX) / 1.5;
  double scaleY = (image.rows - mitY) / 1.5;
  cv::arrowedLine(image, cv::Point(mitX, mitY),
                  cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY), color, 10);
  cv::line(image, cv::Point(mitY, index * 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), index * 20), color, 10);
  double scaleMin = std::min(scaleX, scaleY);
  cv::circle(image, cv::Point2d(mitX, mitY), scaleMin * norm, color, 10);
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

double MVO::calcDisparity(const std::vector<cv::Vec3d> &first, const std::vector<cv::Vec3d> &second)
{
  assert(first.size() == second.size());
  if(first.size() == 0){
    return 0;
  }
  //
  double diff = 0;
  for (auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++)
  {
    diff += cv::norm((*p2) - (*p1));
  }
  diff = diff / first.size();
  //
  return diff;
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

void MVO::setParameters(unsigned int numberOfFeatures, double disparityThreshold){
  _numberOfFeatures = numberOfFeatures;
  _disparityThreshold = disparityThreshold;
}