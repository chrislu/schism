
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "highres_time_stamp.h"

#include <cassert>

#include <boost/cast.hpp>

#include <scm/log.h>

namespace scm {
namespace time {
namespace detail {

high_res_time_stamp::~high_res_time_stamp()
{
}


inline time_duration high_res_time_stamp::get_overhead() const
{
    return (_overhead);
}

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

namespace
{

time_stamp get_pfc_frequency()
{
    LARGE_INTEGER       frequency;

    if (!QueryPerformanceFrequency(&frequency)) {
        char* error_msg;

        FormatMessage(  FORMAT_MESSAGE_IGNORE_INSERTS
                      | FORMAT_MESSAGE_FROM_SYSTEM
                      | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                      0,
                      GetLastError(),
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR)&error_msg,
                      1024,
                      0);

        scm::err() << log::error
                   << "high_res_time_stamp::initialize(): "
                   << "error obtaining performance counter frequency" << log::nline
                   << " - system error message: " << log::nline << "    "
                   << error_msg
                   << log::end;

        LocalFree(error_msg);

        return (false);
    }

    return (boost::numeric_cast<time_stamp>(frequency.QuadPart));
}

time_stamp      pfc_frequency = time_stamp(0);

} // namespace 

high_res_time_stamp::high_res_time_stamp()
    : _overhead(time_duration(0, 0, 0, 0))
{
    pfc_frequency = get_pfc_frequency();
}

bool high_res_time_stamp::initialize()
{
    // calculate overhead of 'now' funktion

    return (true);
}

time_stamp high_res_time_stamp::ticks_per_second()
{
    return (pfc_frequency);
}

time_stamp high_res_time_stamp::now()
{
    static LARGE_INTEGER       current_time_counter;

    if (!QueryPerformanceCounter(&current_time_counter)) {
        char* error_msg;

        FormatMessage(  FORMAT_MESSAGE_IGNORE_INSERTS
                      | FORMAT_MESSAGE_FROM_SYSTEM
                      | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                      0,
                      GetLastError(),
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR)&error_msg,
                      1024,
                      0);

        scm::err() << log::error
                   << "high_res_time_stamp::now(): "
                   << "error obtaining performance counter" << log::nline
                   << " - system error message: " << log::nline << "    "
                   << error_msg
                   << log::end;

        LocalFree(error_msg);

        return (0);
    }

    return (boost::numeric_cast<time_stamp>(current_time_counter.QuadPart));
}

#elif    SCM_PLATFORM == SCM_PLATFORM_LINUX \
      || SCM_PLATFORM == SCM_PLATFORM_APPLE

#include <ctime>

high_res_time_stamp::high_res_time_stamp()
    : _overhead(time_duration(0, 0, 0, 0))
{
}

bool high_res_time_stamp::initialize()
{
    // calculate overhead of 'now' funktion

    return (true);
}

time_stamp high_res_time_stamp::ticks_per_second()
{
    return (1000000000);
}

time_stamp high_res_time_stamp::now()
{
    static timespec current_time;
    
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    return (  boost::numeric_cast<time_stamp>(current_time.tv_sec) * 1000000000
            + boost::numeric_cast<time_stamp>(current_time.tv_nsec));
}

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

} // namespace detail
} // namespace time
} // namespace scm

