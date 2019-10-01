//
// Created by franek on 27.09.19.
//

#include "Utils.hpp"

#ifdef _WIN32
#include <windows.h>
void Utils::SetThreadName(uint32_t dwThreadID, const char* threadName)
{

// DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

THREADNAME_INFO info;
info.dwType = 0x1000;
info.szName = threadName;
info.dwThreadID = dwThreadID;
info.dwFlags = 0;

__try
{
   RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
}
__except(EXCEPTION_EXECUTE_HANDLER)
{
}
}
void Utils::SetThreadName( const char* threadName)
{
 SetThreadName(GetCurrentThreadId(),threadName);
}

void Utils::SetThreadName( std::thread* thread, const char* threadName)
{
 DWORD threadId = ::GetThreadId( static_cast<HANDLE>( thread->native_handle() ) );
 SetThreadName(threadId,threadName);
}

#else
void Utils::SetThreadName(std::thread *thread, const char *threadName) {
  auto handle = thread->native_handle();
  pthread_setname_np(handle, threadName);
}

#include <sys/prctl.h>
void Utils::SetThreadName(const char *threadName) {
  prctl(PR_SET_NAME, threadName, 0, 0, 0);
}
std::string Utils::GetThreadName() {
  auto handle = pthread_self();
  char buffer[100];
  pthread_getname_np(handle,buffer, 100);
  return std::string(buffer);
}

#endif