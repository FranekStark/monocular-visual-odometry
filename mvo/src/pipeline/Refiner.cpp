//
// Created by franek on 25.09.19.
//

#include "Refiner.hpp"

Refiner::Refiner(PipelineStage &precursor,
                 unsigned int out_going_channel_size,
                 IterativeRefinement &iterativeRefinement,
                 unsigned int numberToRefine,
                 unsigned int numberToNote
)
    : PipelineStage(&precursor, out_going_channel_size),
      _iterativeRefinement(iterativeRefinement),
      _frames(numberToNote + 1, nullptr), //One plus, to avoid Overflow!
      _numberToNote(numberToNote),
      _numberToRefine(numberToRefine),
      _baseLine(1) {

  //Currently only 3 available
#ifdef DEBUGIMAGES
  cv::namedWindow("mvocv-RefinerImage", cv::WINDOW_NORMAL);
  cv::moveWindow("mvocv-RefinerImage", 1, 1466);
  cv::resizeWindow("mvocv-RefinerImage", 1164, 454);
  cv::startWindowThread();
#endif
}

Frame *Refiner::stage(Frame *newFrame) {

  //Add One more to keptFrames, cause we have a 'newFrame'.
  _frames.push(newFrame);
  unsigned int keptFrames = _frames.size();
  unsigned int numberToRefine = _numberToRefine;
  unsigned int numberToNote = _numberToNote;

  if (keptFrames < _numberToNote) {
    ROS_WARN_STREAM("not enough Frames to reach 'framesToNote', took as much, as possible!");
    numberToNote = keptFrames;
    numberToRefine = std::min(numberToNote - 1, _numberToRefine); //Takes the biggest restriction
  }

  if (keptFrames < 2) { //Not enough to do anything. -> That case will only be there one Time I think.
    return nullptr;
  }

  //Else we have enough Frames and the vars are set up correctly:

  //Get Data
  std::vector<IterativeRefinement::RefinementFrame> refinementData(numberToNote);
  for (unsigned int i = 0; i < numberToNote; i++) {
    Frame *frame = _frames[(keptFrames - 1) - i];
    refinementData[i].scale = frame->getScaleToPrevious();
    refinementData[i].vec = frame->getBaseLineToPrevious();
    refinementData[i].R = frame->getRotation();
  }


  //Start Refinement
  auto funtolerance = std::pow(10.0, -1 * (newFrame->getParameters().functionTolerance));
  auto gradtolerance = std::pow(10.0, -1 * (newFrame->getParameters().gradientTolerance));
  auto paramtolerance = std::pow(10.0, -1 * (newFrame->getParameters().parameterTolerance));

#ifdef DEBUGIMAGES
  cv::Mat image(newFrame->getImage().size(), CV_8UC3, cv::Scalar(100, 100, 100));
  _iterativeRefinement._debugImage = image;
#endif

  if(newFrame->getParameters().usePreviousScale){
    newFrame->setScaleToPrevious(newFrame->getPreviousFrame(1).getScaleToPrevious());
  }

  _iterativeRefinement.refine(refinementData, numberToRefine, numberToNote, newFrame->getParameters().maxNumThreads,
                              newFrame->getParameters().maxNumIterations,
                              funtolerance,
                              gradtolerance,
                              paramtolerance,
                              newFrame->getParameters().useLossFunction,
                              newFrame->getParameters().lowestLength,
                              newFrame->getParameters().highestLength, *newFrame);

  //Write Back Data:
  for (unsigned int i = 0; i < numberToRefine; i++) {
    Frame *frame = _frames[(keptFrames - 1) - i];
    frame->setScaleToPrevious(refinementData[i].scale);
    frame->setBaseLineToPrevious(refinementData[i].vec);
  }

#ifdef DEBUGIMAGES
  //VisualisationUtils::drawCorrespondences(featureVectors, newFrame->getCameraModel(), image);
  cv::imshow("mvocv-RefinerImage", image);
  cv::waitKey(10);
#endif


  //Enqueue the mostRefined, only if there already enough Frames, to prevent double enqueuing
  //And take the one before the last refined
  if (keptFrames > _numberToRefine) {
    Frame *mostRefined = _frames[keptFrames - (numberToRefine + 1)];
    _baseLine.enqueue({mostRefined->getScaleToPrevious() * mostRefined->getBaseLineToPrevious(),
                       mostRefined->getRotation(), mostRefined->getTimeStamp()
                      });
  }

  //Pass throuh Frames, but Only, if we don't need more

  if (keptFrames >= _numberToNote) { //When that was enough, pass through
    Frame *presFrame = _frames[0];
    _frames.pop();
    return presFrame;
  } else {
    return nullptr;
  }

}

Refiner::~Refiner() {
#ifdef DEBUGIMAGES

  cv::destroyWindow("mvocv-Refiner");
#endif
}
