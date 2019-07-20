#include "SlidingWindow.hpp"

SlidingWindow::SlidingWindow(int len) : _length(len), _firstWindow(nullptr), _lastWindow(nullptr)
{
}

SlidingWindow::~SlidingWindow()
{
}

Window* SlidingWindow::getWindow(int past)
{
  Window* window = _lastWindow;
  int cnt = 0;
  while (window != _firstWindow && cnt != past)
  {
    window = _lastWindow->_windowBefore;
    cnt++;
  }
  return window;
}

void SlidingWindow::newWindow(std::vector<cv::Point2f> trackedFeaturesNow,
                              std::vector<cv::Point2f>& trackedFeaturesBefore, std::vector<unsigned char>& found,
                              cv::Mat image)
{
  (void)(trackedFeaturesBefore);  // Preventing Unused TODO: use?!
  /*Create New Window */
  Window* window = new Window();
  window->_image = image;

  /*Einhaengen */
  window->_windowBefore = _lastWindow;
  _lastWindow = window;

  /*Remove Oldest */
  Window* tmpWindow = _lastWindow;
  for (int i = 0; i <= _length; i++)
  {
    tmpWindow = tmpWindow->_windowBefore;
    if (tmpWindow == nullptr)
    {
      break;
    }
  }
  if (tmpWindow != nullptr)
  {
    delete tmpWindow;
  }

  /*Finding Pairs */
  for (long unsigned int i = 0; i < found.size(); i++)
  {
    if (found[i])
    {
      window->_featuresBefore.insert({ window->_features.size(), i });  // The size is the last (new) Index
      window->_features.push_back(trackedFeaturesNow[i]);
    }  // Else do nothing and Skip these Point
  }
}

void SlidingWindow::addFeaturesToCurrentWindow(std::vector<cv::Point2f> features)
{
  // Put them to the End

  auto const * featuresBefore = this->getFeatures(1);
  if (featuresBefore != nullptr)
  {
    for (auto const & featureFound : features)
    {
      bool found = false;
      for(auto const & featureBefore :  *featuresBefore){
        double dist = cv::norm(featureBefore - featureFound);
        if(dist <= 10){
          found = true;
          break;
        }
      }
      if(!found){
        _lastWindow->_features.push_back(featureFound);
      }

    }
  }else{ //Otherwise add all, because its the first!
    _lastWindow->_features.insert(_lastWindow->_features.end(), features.begin(), features.end());
  }

  // No Values to the Bimap, because these Features are NEW
}

std::vector<cv::Point2f>* SlidingWindow::getFeatures(int past)
{
  Window* window = getWindow(past);
  if (window == nullptr)
  {
    return nullptr;
  }
  return &(window->_features);
}

cv::Mat* SlidingWindow::getImage(int past)
{
  Window* window = this->getWindow(past);
  if (window == nullptr)
  {
    return nullptr;
  }
  return &(window->_image);
}
