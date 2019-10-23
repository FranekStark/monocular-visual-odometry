//
// Created by franek on 25.09.19.
//

#ifndef MVO_SRC_REFINER_HPP_
#define MVO_SRC_REFINER_HPP_

#include "PipelineStage.h"
#include "../algorithms/IterativeRefinement.hpp"
#include "../nils_lib/Ringbuffer.hpp"
#include "../operations/VisualisationUtils.hpp"
#include "OdomData.hpp"
#include "../nils_lib/Ringbuffer.hpp"


class Refiner: public PipelineStage {
 private:

  IterativeRefinement & _iterativeRefinement;
  Frame * _preFrame;
  Frame * _prePreFrame;
  Frame *stage(Frame *newFrame) override;

 public:
  Refiner(PipelineStage &precursor,
          unsigned int out_going_channel_size,
          IterativeRefinement &iterativeRefinement, unsigned int numberToRefine);
  ~Refiner() override;

  Channel<OdomData> _baseLine1;
  Channel<OdomData> _baseLine2;


};

#endif //MVO_SRC_REFINER_HPP_
