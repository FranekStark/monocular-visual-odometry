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
    window = window->_windowBefore;
    cnt++;
  }
  return window;
}

void SlidingWindow::newWindow(const std::vector<cv::Point2f>& trackedFeaturesNow,
                              const std::vector<unsigned char>& found,
                              cv::Mat image)
{
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

void SlidingWindow::addFeaturesToCurrentWindow(std::vector<cv::Point2f> & features)
{
  // Put them to the End
  _lastWindow->_features.insert(_lastWindow->_features.end(), features.begin(), features.end());
  // No Values to the Bimap, because these Features are NEW
}

std::vector<cv::Point2f>& SlidingWindow::getFeatures(int past)
{
  Window* window = getWindow(past);
  assert(window != nullptr);
  return (window->_features);
}

cv::Mat SlidingWindow::getImage(int past)
{
  Window* window = this->getWindow(past);
   assert(window != nullptr);
  return (window->_image);
}

const cv::Vec3d & SlidingWindow::getPosition(int past)
{
  Window* window = this->getWindow(past);
  assert(window != nullptr);
  return (window->_position);
}

const cv::Matx33d & SlidingWindow::getRotation(int past){
  Window* window = this->getWindow(past);
  assert(window != nullptr);
  return (window->_rotation);
}

void SlidingWindow::getCorrespondingFeatures(int window1Index, int window2Index, std::vector<cv::Point2f>& features1,
                                             std::vector<cv::Point2f>& features2)
{
  if (this->getWindow(window1Index) == nullptr || this->getWindow(window2Index) == nullptr)
  {
    return;  // TODO: catch these
  }

  for (unsigned int firstIndex = 0; firstIndex < this->getWindow(window2Index)->_features.size(); firstIndex++)
  {
    int nextIndex = firstIndex;
    bool found = true;
    for (int hist = window2Index; hist < window1Index; hist++)
    {
      Window* thisWindow = this->getWindow(hist);
      const auto nextIndexIt = thisWindow->_featuresBefore.left.find(nextIndex);
      if (nextIndexIt != thisWindow->_featuresBefore.left.end())  // found
      {
        nextIndex = nextIndexIt->second;
      }
      else  // Not Found
      {
        found = false;
        break;  // Next Feature, because History of that Feature not long enough
      }
    }
    if (found)
    {
      features2.push_back(this->getWindow(window2Index)->_features[firstIndex]);
      features1.push_back(this->getWindow(window1Index)->_features[nextIndex]);
    }
  }
}

void SlidingWindow::addTransformationToCurrentWindow(const cv::Vec3d & position,const cv::Matx33d & rotation)
{
  _lastWindow->_rotation = rotation;
  _lastWindow->_position = position;
}

void SlidingWindow::getCorrespondingPosition(int window1Index, int window2Index, cv::Vec3d& position1, cv::Vec3d& position2,
                                             cv::Matx33d& rotation1, cv::Matx33d& rotation2)
{
  Window* window1 = this->getWindow(window1Index);
  Window* window2 = this->getWindow(window2Index);
  assert(window1 != nullptr && window2 != nullptr);

  position1 = window1->_position;
  rotation1 = window1->_rotation;
  position2 = window2->_position;
  rotation2 = window2->_rotation;
}
