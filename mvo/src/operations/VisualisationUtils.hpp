//
// Created by franek on 26.09.19.
//

#ifndef MVO_SRC_VISULISATIONUTILS_HPP_
#define MVO_SRC_VISULISATIONUTILS_HPP_

#define DEBUGIMAGES

#include <image_geometry/pinhole_camera_model.h>
#include "../sliding_window/Frame.hpp"
#include <opencv2/highgui.hpp>

class VisualisationUtils {
 public:
  /**
   * Draws all known Features in specified Frame.
   * If the Feature is tracked (has a preFeature) it will represented by a green point.
   * If its a new Feature (has no preFeature) it will represented by a blue point.
   *
   * @param frame specified frame
   * @param image the image to draw the points to
   */
  static void drawFeatures(const Frame &frame, cv::Mat &image);

  /**
   * Draws the Corresponding (tracked) Features (connected with a Line to an image).
   * The Newest Features will be labelled with '0'. The newest with '0+1' ... Also they will have
   * different colours.
   *
   * @param oldestFrame this Frame has to be a predecessor of the newestFrame
   * @param newestFrame this Frame has to be successors of the oldestFrame
   * @param image the image to draw the Points to
   */
  static void drawCorrespondences(const Frame &oldestFrame, const Frame &newestFrame, cv::Mat &image);

  static void drawCorrespondences(const std::vector<const std::vector<cv::Vec3d> *> vectors,
                                  const image_geometry::PinholeCameraModel &camera,
                                  cv::Mat &image,
                                  const cv::Scalar &lineColor = cv::Scalar(255, 255, 255),
                                  const cv::Scalar &pointColor = cv::Scalar(0, 0, 255));

  /**
   * Draws the Movement as an visualisation to an image.
   *
   * @param frame the Frame which movement to the predecessor has to be visualized
   * @param color the Color of visualisation
   * @param image the image to draw to
   */
  static void drawMovementDebug(const Frame &frame, const cv::Scalar &color, cv::Mat &image, unsigned int index);

  /**
   * Draws the Features of the predecessor frame and this frame. the FEatures of this frame will pe drawed two Times.
   * First as they are and second unrotated to the predecessor. (As if the Camera i only moved but not Rotated)
   *
   * @param featuresBefore this Frame
   * @param featuresNew the image to draw to
   */
  static void drawFeaturesUnrotated(cv::Mat &image,
                                    const std::vector<cv::Point2f> &featuresBefore,
                                    const std::vector<cv::Point2f> &featuresNew,
                                    const std::vector<cv::Point2f> &featuresNewUnrotated);

  /**
   * draws a Rect to specified Image
   * @param image the Imag eto draw to
   * @param rect the rect to daw
   */
  static void drawRect(cv::Mat &image, const cv::Rect &rect);
};

#endif //MVO_SRC_VISULISATIONUTILS_HPP_
