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
#include "../operations/FeatureOperations.h"


class Refiner : public PipelineStage {
 private:
  IterativeRefinement &_iterativeRefinement;
  RingBuffer<Frame *> _frames;
  unsigned int _numberToNote;
  unsigned int _numberToRefine;
#ifdef RATINGDATA
  std::function<void(Rating_Infos, ros::Time)> _ratingCallbackFunction;
#endif

  Frame *stage(Frame *newFrame) override;

 public:
  Refiner(PipelineStage &precursor,
          unsigned int out_going_channel_size,
          IterativeRefinement &iterativeRefinement,
          unsigned int numberToRefine,
#ifdef RATINGDATA
          std::function<void(Rating_Infos, ros::Time)> ratingCallbackFunction,
#endif
          unsigned int numberToNote
  );
  ~Refiner() override;

  Channel<OdomData> _baseLine;

};

#endif //MVO_SRC_REFINER_HPP_
