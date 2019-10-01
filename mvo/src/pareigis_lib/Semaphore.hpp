/*
 * Semaphore.hpp
 *
 *  Created on: 20.10.2017
 *      Author: Nils Schoenherr
 */

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#pragma once

#include <mutex>
#include <condition_variable>


// the semaphore class should be in its own .h file. See the semaphore post on this site
class semaphore {
 public:
  semaphore(const size_t count = 0)
      : count_(count)
      , condition_()
      , mtx_()
  {}

  semaphore(const semaphore&) = delete;
  semaphore& operator=(const semaphore&) = delete;
  semaphore(semaphore&&) = delete;
  semaphore& operator=(semaphore&&) = delete;

  inline void post(void) {
    std::lock_guard<std::mutex> lock(mtx_);
    count_++;
    condition_.notify_one();
  }

  inline void wait(void) {
    std::unique_lock<std::mutex> lock(mtx_);
    condition_.wait(lock, [&]{return count_>0;});
    count_--;
  }

  inline bool try_wait(void) {
    std::unique_lock<std::mutex> lock(mtx_);
    if(count_ > 0) {
      count_--;
      return true;
    }
    return false;
  }

  inline size_t get_value() const {
    return count_;
  }

  void destroy() {
    count_=10;
    condition_.notify_all();
  }
 private:
  size_t count_;
  std::condition_variable condition_;
  std::mutex mtx_;
};

#endif /* SEMAPHORE_HPP_ */