#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), _frameCounter(0), _blockSize(2), _apertureSize(3), _k(0.04), _thresh(200)
{
  cv::namedWindow("original", cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow("cornerImage", cv::WINDOW_GUI_EXPANDED);
}

MVO::~MVO()
{
  cv::destroyWindow("original");
  cv::destroyWindow("cornerImage");
}

void MVO::handleImage(const cv::Mat &image, const image_geometry::PinholeCameraModel &cameraModel)
{
  // TODO: Pointer to CV shared Pointers
  /*Original */
  cv::imshow("original", image);
  /*Grayscale */
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BayerBG2GRAY);

  /*Track Features */
  std::vector<cv::Point2f> trackedFeatures;
  std::vector<unsigned char> found;
  cv::Mat *prevImage = _slidingWindow.getImage(0);
  cv::Mat prevB = _slidingWindow.getPosition(0);
  std::vector<cv::Point2f> *prevFeatures = _slidingWindow.getFeatures(0);
  if (prevImage != nullptr)
  {  // Otherwise it is the first Frame
    this->trackFeatures(grayImage, *prevImage, *prevFeatures, trackedFeatures, found);
  }

  /*New Window-Frame */
  _slidingWindow.newWindow(trackedFeatures, *prevFeatures, found, grayImage);

  /*NewFeatures */
  std::vector<cv::Point2f> newFeatures = this->detectCorners(grayImage, 20);
  _slidingWindow.addFeaturesToCurrentWindow(newFeatures);

  /*Mark Features*/
  cv::Mat cornerImage = image.clone();
  std::stringstream text;
  text << "Number of Features: " << _slidingWindow.getFeatures(0)->size();
  cv::putText(cornerImage, text.str(), cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255));
  for (auto const &corner : *(_slidingWindow.getFeatures(0)))
  {
    cv::circle(cornerImage, cv::Point(corner), 10, cv::Scalar(0, 0, 255), -10);
  }

  if (_frameCounter > 0)
  {                                                             // Otherwise it is the first Frame
                                                                /*Calc BaseLine */
    std::vector<cv::Point2f> thisFeaturesCV, beforeFeaturesCV;  // TODO: Avoid the Conversion and CV to Double!
    _slidingWindow.getCorrespondingFeatures(1, 0, beforeFeaturesCV, thisFeaturesCV);
    std::vector<Eigen::Vector2d> thisFeatures(thisFeaturesCV.size());
    std::vector<Eigen::Vector2d> beforeFeatures(beforeFeaturesCV.size());
    for (unsigned i = 0; i < thisFeaturesCV.size(); i++)
    {
      // euclid Norm:
      const cv::Point3d beforeFeatureCV = cameraModel.projectPixelTo3dRay(beforeFeaturesCV[i]);
      const cv::Point3d thisFeatureCV = cameraModel.projectPixelTo3dRay(thisFeaturesCV[i]);

      thisFeatures[i] = Eigen::Vector2d(thisFeatureCV.x, thisFeatureCV.y);
      beforeFeatures[i] = Eigen::Vector2d(beforeFeatureCV.x, beforeFeatureCV.y);
    }

    Eigen::Translation3d translation =
        this->calculateBaseLineMLESAC(thisFeatures, beforeFeatures, Eigen::Quaterniond());

    if (_frameCounter == 1)
    {
      std::vector<double> depth;
      int value;
      this->reconstructDepth(depth, thisFeatures, beforeFeatures, Eigen::Quaterniond(),
                             translation);  // TODO:: hier nur Inlier //TODO: rot
      for (std::vector<double>::iterator it = depth.begin(); it < depth.end(); it++)
      {
        if ((*it) < 0)
        {
          value--;
        }
        else if ((*it) > 0)
        {
          value++;
        }
      }

      cv::Mat b(3, 1, CV_64F);
      b.at<double>(0, 0) = translation.x();
      b.at<double>(0, 1) = translation.y();
      b.at<double>(0, 2) = translation.z();
      if (value < 0)
      {
        b = b * -1;
      }
      _slidingWindow.addTransformationToCurrentWindow(b, cv::Mat::eye(3, 3,CV_64F));
      prevB = b;
    }
    else
    {  // Scale estimation (Iterative Refinement)
      std::vector<cv::Point2f> thisFeatures, beforeFeatures;
      cv::Mat thisRotation, beforeRotation;
      cv::Mat thisPosition, beforePosition;

      _slidingWindow.getCorrespondingFeatures(2, 0, beforeFeatures, thisFeatures);
      _slidingWindow.getCorrespondingPosition(2, 0, beforePosition, thisPosition, beforeRotation, thisRotation);

      thisRotation = cv::Mat::eye(3, 3,CV_64F);  // TODO: REAL ROTATION
      thisPosition = cv::Mat(3, 1, CV_64F);
      thisPosition.at<double>(0) = translation.x();
      thisPosition.at<double>(1) = translation.y();
      thisPosition.at<double>(2) = translation.z();

      std::vector<IterativeRefinement::Input> input;

      for (unsigned int i = 0; i < thisFeatures.size(); i++)
      {
        const cv::Point3d beforeFeature = cameraModel.projectPixelTo3dRay(beforeFeatures[i]);
        const cv::Point3d thisFeature = cameraModel.projectPixelTo3dRay(thisFeatures[i]);

        IterativeRefinement::Input in = { cv::Mat(thisFeature), thisRotation, cv::Mat(beforeFeature), beforeRotation,
                                          beforePosition };
        input.push_back(in);
      }

      IterativeRefinement::GaussNewton(IterativeRefinement::Func, input, thisPosition);

      prevB = prevB + thisPosition;
      _slidingWindow.addTransformationToCurrentWindow(prevB, cv::Mat::eye(3, 3,CV_64F));

      ROS_INFO_STREAM("Scaled/Refined Translation: " << std::endl << thisPosition);
    }

    ROS_INFO_STREAM("Translation: " << std::endl
                                    << translation.x() << std::endl
                                    << translation.y() << std::endl
                                    << translation.z() << std::endl);

    /**
     * Draw Base-Line:
     */
    int mitX = double(cornerImage.cols) / 2.0;
    int mitY = double(cornerImage.rows) / 2.0;
    double scaleX = (cornerImage.cols - mitX) / 1.5;
    double scaleY = (cornerImage.rows - mitY) / 1.5;
    cv::arrowedLine(cornerImage, cv::Point(mitX, mitY),
                    cv::Point((scaleX * translation.x()) + mitX, (scaleY * translation.y()) + mitY),
                    cv::Scalar(0, 255, 0), 10);
    cv::line(cornerImage, cv::Point(mitY, 20), cv::Point(mitY + (scaleX * translation.z()), 20), cv::Scalar(0, 255, 0),
             10);
  }
  else if (_frameCounter == 0)
  {
    _slidingWindow.addTransformationToCurrentWindow(cv::Mat::zeros(3, 1, CV_64F), cv::Mat::eye(3, 3,CV_64F));
    prevB = cv::Mat::zeros(3, 1, CV_64F);
  }
  imshow("cornerImage", cornerImage);
  cv::waitKey(1);
  _frameCounter++;
}

// Must be Grayscale
std::vector<cv::Point2f> MVO::detectCorners(const cv::Mat &image, int num)
{
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(image, corners, num, double(0.01), double(10.0), cv::noArray(), _blockSize, bool(true),
                          _k);  // Corners berechnen TODO: More params

  // Subpixel-genau:
  if (corners.size() > 0)
  {
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria =
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(image, corners, winSize, zeroZone, criteria);
  }
  return corners;
  std::vector<cv::Point2f> result;
}

void MVO::setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh)
{
  // TODO: Concurrent, when using more Threads
  _blockSize = blockSize;
  _apertureSize = aperatureSize;
  _k = k;
  _thresh = thresh;
}

void MVO::trackFeatures(const cv::Mat &nowImage, const cv::Mat &prevImage, const std::vector<cv::Point2f> &prevFeatures,
                        std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found)
{
  std::vector<cv::Mat> nowPyramide;
  std::vector<cv::Mat> prevPyramide;
  cv::Size winSize(21, 21);  // Has to be the same as in calcOpeitcalcOpticalFLow
  int maxLevel = 3;
  // TODO: The call of these Funtions only make sense, if we store the pyramids for reuse.
  // Otherwise calcOpticalFlow could to this on its own.
  cv::buildOpticalFlowPyramid(nowImage, nowPyramide, winSize, maxLevel);
  cv::buildOpticalFlowPyramid(prevImage, prevPyramide, winSize, maxLevel);

  // TODO: Handle Case, when no before Features!
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(prevPyramide, nowPyramide, prevFeatures, trackedFeatures, found, error);  // TODO: more
                                                                                                     // Params
}

Eigen::Translation3d MVO::calculateBaseLine(const std::vector<Eigen::Vector2d> &mt,
                                            const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rhi)
{
  (void)(rhi);
  // TODO: smae size vor mt and mhi check

  // DIESE METHODE NIMMT NUR 3

  Eigen::Matrix<double, 3, 3> A(3, 3);  // TODO: Row wise

  for (unsigned int i = 0; i < 3;
       i++)  // TODO: Hier werden nur müssten 3 genommen, sonst könnte die Determinante nicht geprüft werden
  {
    // TODO: Unrotate

    double x1 = mhi[i][0];
    double x2 = mt[i][0];
    double y1 = mhi[i][1];
    double y2 = mt[i][1];

    double a1 = -(y2 - y1);
    double a2 = (x2 - x1);
    double a3 = ((y2 - y1) * x2 - (x2 - x1) * y2);

    A(i, 0) = a1;
    A(i, 1) = a2;
    A(i, 2) = a3;
  }

  if (A.rows() > 3 || A.determinant() != 0)  // TODO: eigentlich  Keine Translation, wenn A singular ist
  {
    // A * x = 0 --> SVD
    Eigen::BDCSVD<Eigen::Matrix<double, Eigen::Dynamic, 3>> svd(A,
                                                                Eigen::DecompositionOptions::ComputeFullV);  // TODO:
                                                                                                             // Thin or
                                                                                                             // full
    auto &V = svd.matrixV();

    // Last Column of V
    Eigen::Vector3d lastColumn = V.col(V.cols() - 1);

    Eigen::Translation3d translation(lastColumn);

    return translation;
  }
  else
  {
    // Eine Determinante ist gleich null, wenn

    //-eine Zeile/Spalte nur aus Nullen besteht
    //-zwei Zeilen/Spalten gleich sind
    //-eine Zeile/Spalte eine Linearkombination anderer Zeilen/Spalten ist

    return Eigen::Translation3d(0, 0, 0);
  }
}

Eigen::Translation3d MVO::calculateBaseLineMLESAC(const std::vector<Eigen::Vector2d> &mt,
                                                  const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh)
{
  // TODO: from: https://inside.mines.edu/~whoff/courses/EENG512/lectures/20-Ransac.pdf

  // unsigned int max_iterations = 500;

  const double threshold = cos(3 * PI / 180.0);  // Threshold Laut Paper zwischen cos(7°) und cos(3°) setzten

  unsigned int N = mt.size();  // Anzahl aller Feature-Paare
  unsigned int s = 3;          // Größe der Subsets, für die Berechnung
  double Ps = 0.99;  // Gwünschte Wahrscheinlichkeit, dass das am Ende gewählte Feature-Paar-Set keine Outlier enthält.

  double inlierProbability = 0.3;  // FIRST GUESS (für die Wahrscheinlichekit, dass ein Feature-Paar ein inlier ist),
                                   // der sehr schlecht ist! (Wird später im Algorithmus noch angepasst)

  /*Erster Estimate der Benötigten Iterationen ---->*/
  unsigned int nInlier = N * inlierProbability;  // Anzahl der Inlier im Datensatz für die inlierProbability
  unsigned int m = boost::math::binomial_coefficient<double>(nInlier, s);  // n über k -> 3 aus N -> Alle
                                                                           // Möglichkeiten 3 aus Inlier zu nehmen
  unsigned int n = boost::math::binomial_coefficient<double>(N, s);        // Alle Möglichkeiten 3 aus allen
                                                                           // Feature-Paaren zu Nehmen
  double p = double(m) / double(n);  // Wahrscheinlichkeit, dass in einem Sample de größe s (3), alles Inlier sind
  unsigned int nIterations = ceil(log(1 - Ps) / log(1 - p));
  ROS_INFO_STREAM("Initial erwartete Anzahl von Iterationen: " << nIterations << std::endl);
  /* <----- */

  std::random_device randomDevice;
  std::uniform_int_distribution<unsigned int> randomDist(0, N);
  std::mt19937 randomGenerator(randomDevice());  // TODO: ist das die beste Möglichkeit für Zufallswerte?

  unsigned int iteration = 0;
  double pBest = -1;                   // Beste Wahrscheinlichkeit, die bisher gefunden wurde
  std::vector<int> bestInLierIndexes;  // Indices, der Inlier des Besten Sets
  int nBest = 0;                       // Anzahl der besten

  while (iteration < nIterations)
  {
    /*Subsets erstellen */
    std::vector<Eigen::Vector2d> mtSet;  // TODO: References?
    std::vector<Eigen::Vector2d> mhiSet;

    for (unsigned i = 0; i < s; i++)
    {
      unsigned int randomIndex = randomDist(randomGenerator);
      mtSet.push_back(mt[randomIndex]);
      mhiSet.push_back(mhi[randomIndex]);
    }

    Eigen::Translation3d translation = calculateBaseLine(mtSet, mhiSet, rh);  // Berechnung aus dem Subsample

    iteration++;

    Eigen::Vector3d b(translation.x(), translation.y(), translation.z());
    /*Calculate Penalty and Get Inliers*/
    nInlier = 0;  // Neu Zählen, wie viele wir haben
    std::vector<int> inLierIndexes;
    double nProbability = 0;  // Gesamtwahrscheinlichkeit dieses Datensatzes
    inLierIndexes.clear();
    for (unsigned int i = 0; i < N; i++)
    {
      Eigen::Vector3d m2 = mt[i].homogeneous();   // To Homgenous
      Eigen::Vector3d m1 = mhi[i].homogeneous();  // To Homogenous
      // TODO: Unrotate m2

      Eigen::Matrix3d I;
      I << 1, 0, 0, 0, 1, 0, 0, 0, 1;

      double pInlier =  // Probability of that Inlier.
          (m2.transpose() * (I - (b * b.transpose())) * m1)(0) /
          //---------------------------------------------------------------------------------------
          (sqrt(m1.squaredNorm() - pow((m1.transpose() * b)(0), 2)) *
           sqrt(m2.squaredNorm() - pow((m2.transpose() * b)(0), 2)));
      if (pInlier > threshold)
      {  // Juhu, there is an Inlier :)
        nInlier++;
        nProbability += pInlier;
        inLierIndexes.push_back(i);
      }
    }

    /*Update der Anzahl der benötigten Iterationen */
    if (double(nInlier) / double(N) > inlierProbability)
    {
      inlierProbability = double(nInlier) / double(N);
      unsigned int m = boost::math::binomial_coefficient<double>(nInlier, s);  // n über k -> 3 aus N ->
                                                                               // Alle Möglichkeiten 3
                                                                               // aus Inlier zu nehmen
      unsigned int n = boost::math::binomial_coefficient<double>(N, s);        // Alle Möglichkeiten 3 aus allen
                                                                               // Feature-Paaren zu Nehmen
      double p = double(m) / double(n);  // Wahrscheinlichkeit, dass in einem Sample de größe s (3), alles Inlier sind
      nIterations = ceil(log(1 - Ps) / log(1 - p));
      ROS_INFO_STREAM("Iteration " << iteration << ": Neue erwartete Anzahl von Iterationen: " << nIterations
                                   << std::endl);
    }

    if (nProbability > pBest)
    {  // Neuer bester wurde gefunden
      pBest = nProbability;
      nBest = nInlier;
      bestInLierIndexes = inLierIndexes;
      ROS_INFO_STREAM("Iteration " << iteration << ": Neues Bestes Modell mit " << nInlier << " Inliern" << std::endl);
    }
  }

  // Ende des Algorithmus, finale Berechnung über den Besten:
  std::vector<Eigen::Vector2d> mtSet;  // TODO: References?
  std::vector<Eigen::Vector2d> mhiSet;
  for (int index : bestInLierIndexes)
  {
    mtSet.push_back(mt[index]);
    mhiSet.push_back(mhi[index]);
  }
  ROS_INFO_STREAM("Iteration " << iteration << ": Bestes Modell mit " << nBest << " und Rank: " << pBest << std::endl);

  Eigen::Translation3d baseLine = calculateBaseLine(mtSet, mhiSet, rh);

  return baseLine;
}

void MVO::reconstructDepth(std::vector<double> &depth, const std::vector<Eigen::Vector2d> &m2L,
                           const std::vector<Eigen::Vector2d> &m1L, const Eigen::Quaterniond &r,
                           const Eigen::Translation3d &b)
{
  // TODO: ROTATION
  (void)(r);
  const std::vector<Eigen::Vector2d>::const_iterator m1, m2;
  ROS_INFO_STREAM("depth: ");
  for (unsigned int i = 0; i < m2L.size(); i++)
  {
    Eigen::Vector3d m2 = m2L[i].homogeneous();  // To Homgenous
    Eigen::Vector3d m1 = m1L[i].homogeneous();  // To Homogenous
    Eigen::Matrix3d C;
    C << 1, 0, -m2.x(),  //
        0, 1, -m2.y(),   //
        -m2.x(), -m2.y(), m2.x() * m2.x() + m2.y() * m2.y();
    double Z = (m1.transpose() * C * b.vector())(0) / (m1.transpose() * C * (m1))(0);
    ROS_INFO_STREAM(Z << ", ");
    depth.push_back(Z);
  }
  ROS_INFO_STREAM(std::endl);
}