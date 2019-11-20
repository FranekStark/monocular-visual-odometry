//
// Created by franek on 26.09.19.
//

#include "PipeLineEnd.hpp"
PipeLineEnd::PipeLineEnd(PipelineStage &precursor) : PipelineStage(&precursor,
                                                                   1),
                                                     _lastFrame(nullptr) {}
Frame *PipeLineEnd::stage(Frame *newFrame) {
  if (_lastFrame != nullptr) { //Only if there is a Last Frame
#ifdef RATINGDATA

#endif
    printRatingData(_lastFrame);
    LOG_DEBUG("Deleted end Frame: " << _lastFrame);
    //Delete kept Frame
    delete _lastFrame;
  }
  //Keep the new One, cause the preStages could need the Data
  _lastFrame = newFrame;
  return nullptr;
}

#ifdef RATINGDATA
void PipeLineEnd::printRatingData(const Frame *frame) {
  std::stringstream out;
  out << std::endl;
  out << "=====================================================================" << std::endl;
  out << "=================================END=================================" << std::endl;
  out << "=====================================================================" << std::endl;
  out << "Frame: " << frame << " at: " << frame->getTimeStamp().toSec() << std::endl;
  out << "---------------------------------------------------------------------" << std::endl;
  out << "RANSAC_outsortet_features: " << frame->_infos.RANSAC_outsortet_features << std::endl;
  out << "RANSAC_probability: " << frame->_infos.RANSAC_probability << std::endl;
  out << "MERGER_disparity: " << frame->_infos.MERGER_disparity << std::endl;
  out << "ESTIMATED_scale_baseline:  " << frame->_infos.ESTIMATED_scale << " \t * \t" <<  frame->_infos.ESTIMATED_baseline << std::endl;
  assert(frame->_infos.REFINED_baselines.size() == frame->_infos.REFINED_scales.size());
  for(unsigned int i = 0; i < frame->_infos.REFINED_baselines.size(); i++){
    out << "REFINED_scale_baseline[" << i << "]: "<< frame->_infos.REFINED_scales[i] << " \t * \t" <<  frame->_infos.REFINED_baselines[i] << std::endl;
  }
  out << "=====================================================================" << std::endl;
  out << "=====================================================================" << std::endl;
  out << "=====================================================================" << std::endl;
  ROS_INFO_STREAM(out.str());
}

#endif
