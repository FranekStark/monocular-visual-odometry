//
// Created by franek on 25.09.19.
//

#ifndef MVO_SRC_REFINER_HPP_
#define MVO_SRC_REFINER_HPP_

#include "PipelineStage.h"
#include "IterativeRefinement.hpp"
#include "nils_lib/Ringbuffer.hpp"

class Refiner: PipelineStage {
 private:

  IterativeRefinement & _iterativeRefinement;
  RingBuffer<Frame *> _ringBuffer;

 public:
  Refiner(SlidingWindow &sliding_window,
          PipelineStage &precursor,
          unsigned int out_going_channel_size,
          IterativeRefinement &iterativeRefinement, unsigned int numberToRefine);
  ~Refiner();
 private:
  Frame *stage(Frame *newFrame) override;

};

#endif //MVO_SRC_REFINER_HPP_
