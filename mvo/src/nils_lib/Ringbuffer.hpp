/*
 * RingBuffer.hpp
 *
 *  Created on: 19.09.2019
 *      Author: Nils Schoenherr
 */

#ifndef RINGBUFFER_HPP
#define RINGBUFFER_HPP

#include <vector>

template<typename T>
class RingBuffer {
  std::vector<T> buffer;
  int size_;
  int cnt_;
 public:
  int writePos;
  int readPos;

 public:

  /**
   * @brief Constructor
   * @param size - size of the buffer
   **/
  explicit RingBuffer(int size)
      : buffer(size), size_(size), cnt_(0), writePos(0), readPos(0) {

  }

  /**
  * @brief Constructor
  * @param size - size of the buffer
  * @param value - the value to init each element with
  **/
  explicit RingBuffer(int size, const T &value)
      : buffer(size, value), size_(size), cnt_(0), writePos(0), readPos(0) {

  }

  /**
   * @brief checks the number of elements currently in the Buffer
   * @return number of items in buffer
   **/
  unsigned int size() {
    return (size_ + writePos - readPos) % size_;
  }

  /**
   * @brief pushes one item to buffer
   * @param c - next item
   **/
  void push(T c) {
    buffer[writePos] = c;
    writePos = (writePos + 1) % size_;
  }

  /**
   * @brief pops items from the buffer
   * @param count - number of items to pop
   **/
  void pop(unsigned int count = 1) {
    if (count > size()) {
      readPos = writePos;
    } else {
      readPos = (readPos + count) % size_;
    }
  }

  /**
   * @brief retrieves the object on position pos in the RingBuffer
   * @param pos - position in buffer relative to readPosition
   * @return char - the char at the pos
   **/
  T operator[](int pos) {
    return buffer[(readPos + pos) % size_];
  }
};

#endif //RINGBUFFER_HPP