/*
 * Channel.hpp
 *
 *  Created on: 20.10.2017
 *      Author: Nils Schoenherr
 */

#ifndef CHANNEL_HPP_
#define CHANNEL_HPP_

#include "Semaphore.hpp"
#include <queue>


class ChannelException: public std::exception
{
  virtual const char* what() const throw(){
    return "Channel exception happened\n";
  }
};

template <typename Type_>
class Channel {
 public:
  explicit Channel(const size_t max_size)
      : sem_free_spaces_{max_size}
      , sem_size_{}
      , queue_{}
      , mtx_{}
      , destroyed(false)
  {}

  size_t size(void) const {
    return sem_size_.get_value();
  }

  size_t max_size(void) const {
    return sem_free_spaces_.get_value() + sem_size_.get_value();
  }

  inline Type_ dequeue(void){
    sem_size_.wait();
    if (destroyed) {
      throw(ChannelException());
    }
    mtx_.lock();
    auto return__ = queue_.front();
    queue_.pop();
    mtx_.unlock();
    sem_free_spaces_.post();
    return return__;
  }

  inline void enqueue(const Type_ element){
    sem_free_spaces_.wait();
    if (destroyed) {
      throw(ChannelException());
    }
    mtx_.lock();
    queue_.push(element);
    mtx_.unlock();
    sem_size_.post();
  }

  void operator<<(const Type_ element){
    try {
      enqueue(element);
    } catch (ChannelException& ex){
      throw(ex);
    }
  }

  Type_ friend operator<<(Type_& target, Channel<Type_>& source) {
    return target = source.dequeue();
  }

  void friend operator<<(Channel<Type_>& target,
                         Channel<Type_>& source) {
    target.enqueue(source.dequeue());
  }

  void destroy() {
    destroyed = true;

    sem_free_spaces_.post();
    sem_size_.post();

    sem_free_spaces_.destroy();
    sem_size_.destroy();
  }
 private:
  semaphore sem_free_spaces_, sem_size_;
  std::queue<Type_> queue_;
  std::mutex mtx_;
  bool destroyed;
};

#endif /* CHANNEL_HPP_ */