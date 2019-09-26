//
// Created by franek on 25.09.19.
//

#ifndef MVO_SRC_REFINER_HPP_
#define MVO_SRC_REFINER_HPP_

#include "PipelineStage.h"
#include "../algorithms/IterativeRefinement.hpp"
#include "../nils_lib/Ringbuffer.hpp"
#include "../operations/VisualisationUtils.hpp"
#include "../mvo.hpp"

class Refiner: PipelineStage {
 private:

  IterativeRefinement & _iterativeRefinement;
  RingBuffer<Frame *> _ringBuffer;
  Frame *stage(Frame *newFrame) override;

 public:
  Refiner(PipelineStage &precursor,
          unsigned int out_going_channel_size,
          IterativeRefinement &iterativeRefinement, unsigned int numberToRefine);
  ~Refiner();

  Channel _baseLine;


};

#endif //MVO_SRC_REFINER_HPP_
