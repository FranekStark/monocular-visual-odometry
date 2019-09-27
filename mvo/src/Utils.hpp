//
// Created by franek on 27.09.19.
//

#ifndef MVO_SRC_UTILS_HPP_
#define MVO_SRC_UTILS_HPP_

#include <ros/ros.h>
#include <thread>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

#define LOG_DEBUG(args) ROS_DEBUG_STREAM("(" << Utils::GetThreadName() << " :" << __PRETTY_FUNCTION__ << ":" <<__func__<<":"<< __LINE__ << ") : " << args)

class Utils {
 public:
#ifdef _WIN32
  const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
    typedef struct tagTHREADNAME_INFO
    {
       DWORD dwType; // Must be 0x1000.
       LPCSTR szName; // Pointer to name (in user addr space).
       DWORD dwThreadID; // Thread ID (-1=caller thread).
       DWORD dwFlags; // Reserved for future use, must be zero.
    } THREADNAME_INFO;
#pragma pack(pop)

  static void SetThreadName(uint32_t dwThreadID, const char* threadName);
#else
  static std::string GetThreadName();
#endif
  static void SetThreadName(std::thread *thread, const char *threadName);
  static void SetThreadName(const char *threadName);
};

#endif //MVO_SRC_UTILS_HPP_
