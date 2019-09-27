//
// Created by franek on 27.09.19.
//
#include "Frame.hpp"

void Frame::lock() const{
  LOG_DEBUG("tries to lock Frame" << this);
  _lock.lock();
  LOG_DEBUG("locked Frame" << this);
}

void Frame::unlock() const{
  LOG_DEBUG("unlock Frame" << this);
  _lock.unlock();
}
