#include "EpipolarGeometry.hpp"

#include <ros/ros.h>

EpipolarGeometry::EpipolarGeometry(image_geometry::PinholeCameraModel cameraModel):
_cameraModel(cameraModel),
PI(3.14159265),
THRESHOLD(cos(3*PI/180.0)),
Ps(0.99),
_randomGenerator(_randomDevice())
{
    _cameraModel = cameraModel;
}

EpipolarGeometry::~EpipolarGeometry()
{
}

unsigned int EpipolarGeometry::estimateNumberOfIteration(unsigned int N, double inlierProbability, unsigned int s){
  unsigned int nInlier = N * inlierProbability;  // Anzahl der Inlier im Datensatz für die inlierProbability
  unsigned int m = boost::math::binomial_coefficient<double>(nInlier, s);  // n über k -> 3 aus N -> Alle
                                                                           // Möglichkeiten 3 aus Inlier zu nehmen
  unsigned int n = boost::math::binomial_coefficient<double>(N, s);        // Alle Möglichkeiten 3 aus allen
                                                                           // Feature-Paaren zu Nehmen
  double p = double(m) / double(n);  // Wahrscheinlichkeit, dass in einem Sample de größe s (3), alles Inlier sind
  unsigned int nIterations = ceil(log(1 - Ps) / log(1 - p));
  ROS_INFO_STREAM("Initial erwartete Anzahl von Iterationen: " << nIterations << std::endl);
  return nIterations;
}

unsigned int EpipolarGeometry::reEstimateNumberOfIteration(unsigned int N, unsigned int nInlier, unsigned int s){
      double inlierProbability = double(nInlier) / double(N);
      unsigned int m = boost::math::binomial_coefficient<double>(nInlier, s);  // n über k -> 3 aus N ->
                                                                               // Alle Möglichkeiten 3
                                                                               // aus Inlier zu nehmen
      unsigned int n = boost::math::binomial_coefficient<double>(N, s);        // Alle Möglichkeiten 3 aus allen
                                                                               // Feature-Paaren zu Nehmen
      double p = double(m) / double(n);  // Wahrscheinlichkeit, dass in einem Sample de größe s (3), alles Inlier sind
      unsigned int nIterations = ceil(log(1 - Ps) / log(1 - p));
      ROS_INFO_STREAM("Iteration " << iteration << ": Neue erwartete Anzahl von Iterationen: " << nIterations
                                   << std::endl);
    return nIterations;
}

cv::Vec3d EpipolarGeometry::estimateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt, const cv::Matx33d &rhi){
  assert(mhi.size() == mt.size());

  unsigned int N = mt.size();  // Anzahl aller Feature-Paare
  unsigned int s = 3;          // Größe der Subsets, für die Berechnung

  double inlierProbability = 0.3;  // FIRST GUESS (für die Wahrscheinlichekit, dass ein Feature-Paar ein inlier ist),
                                   // der sehr schlecht ist! (Wird später im Algorithmus noch angepasst)

  unsigned int nIterations = this->estimateNumberOfIteration(N, inlierProbability, s);

  std::uniform_int_distribution<unsigned int> randomDist(0, N);
   

  unsigned int iteration = 0;
  double pBest = -1;                   // Beste Wahrscheinlichkeit, die bisher gefunden wurde
  std::vector<int> bestInLierIndexes;  // Indices, der Inlier des Besten Sets
  int nBest = 0;                       // Anzahl der besten

  while (iteration < nIterations)
  {
    /*Subsets erstellen */
    std::vector<cv::Vec3d> mtSet; 
    std::vector<cv::Vec3d> mhiSet;

    for (unsigned i = 0; i < s; i++)
    {
      unsigned int randomIndex = randomDist(_randomGenerator);
      mtSet.push_back(mt[randomIndex]);
      mhiSet.push_back(mhi[randomIndex]);
    }

    cv::Vec3d b = this->calculateBaseLine(mtSet, mhiSet, rh);  // Berechnung aus dem Subsample
    iteration++;
    
    /*Calculate Penalty and Get Inliers*/
    unsigned int nInlier = 0;  // Neu Zählen, wie viele wir haben
    std::vector<int> inLierIndexes;
    double nProbability = 0;  // Gesamtwahrscheinlichkeit dieses Datensatzes
    inLierIndexes.clear();
    for (auto m1 = mhi.begin(), m2 = mt.begin(); m1 != mhi.end() && m2 != mt.end(); m1++, m2++)
    {
      // TODO: Unrotate m2
      (void)(rhi);

      cv::Matx33d I = cv::Matx33d::eye();
      double pInlier =  // Probability of that Inlier.
          (m2->t() * (I - (b * b.t())) * (*m1))(0) /
          //---------------------------------------------------------------------------------------
          (sqrt(cv::norm(*m1,cv::NORM_L2SQR) - cv::pow((m1->t() * b)(0), 2)) *
           sqrt(cv::norm(*m2,cv::NORM_L2SQR) - cv::pow((m2->t() * b)(0), 2)));

      if (pInlier > THRESHOLD)
      {  // Juhu, there is an Inlier :)
        nInlier++;
        nProbability += pInlier;
        inLierIndexes.push_back(std::distance(mhi.begin(), m1));
      }
    }

    /*Update der Anzahl der benötigten Iterationen */
    if (double(nInlier) / double(N) > inlierProbability)
    {

      nIterations = this->reEstimateNumberOfIteration(N, nInlier, s);
     
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
cv::Vec3d EpipolarGeometry::calculateBaseLine(const std::vector<cv::Vec3d> &mhi, const std::vector<cv::Vec3d> &mt, const cv::Matx33d &rhi){
    assert(mhi.size() == mt.size());
    
    //A*b = 0
    
   
    cv::Mat A = cv::Mat::zeros(mhi.size(), 3, CV_64F);  //Row, Col
    cv::Vec3d b;
    
    for(auto m1 = mhi.begin(), m2 = mt.begin(); m1 != mhi.end() && m2 != mt.end(); m1++, m2++){
        double a1 = -((*m2)(1) - (*m1)(1));
        double a2 = ((*m2)(0) - (*m1)(0));
        double a3 = (((*m2)(1) - (*m1)(1)) * (*m2)(1) - ((*m2)(0) - (*m1)(0))*(*m2)(1));

        //TODO: Unrotate!
        (void)(rhi);
        int i = std::distance(mhi.begin(), m1);
        A.at<double>(i, 0) = a1;
        A.at<double>(i, 1) = a2;
        A.at<double>(i, 2) = a3;
    }

    if(A.rows > 3 || cv::determinant(A) != 0){ //TODO: Hier muss was passieren, aber was?
      cv::Mat w, u, vt;
      
      cv::SVD::compute(A, w, u, vt, cv::SVD::Flags::MODIFY_A | cv::SVD::Flags::FULL_UV); //Full, because we need V  
      cv::Mat lastCol =  vt.t().col(vt.t().size().width - 1); //Last Row of V is our Result
      b(0) = lastCol.at<double>(0,0);
      b(1) = lastCol.at<double>(1,0);
      b(2) = lastCol.at<double>(2,0); //TODO: Ugly conversion
    }
    return b;
}

