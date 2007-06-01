
#include "highres_time_stamp.h"

#include <cassert>

#include <boost/cast.hpp>

#include <scm_core/console.h>
#include <scm_core/platform/platform.h>

namespace scm {
namespace time {
namespace detail {

high_res_time_stamp::high_res_time_stamp()
    : _overhead(time_duration(0, 0, 0, 0))
{
}

high_res_time_stamp::~high_res_time_stamp()
{
}


inline time_duration high_res_time_stamp::get_overhead() const
{
    return (_overhead);
}

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <windows.h>

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

        console.get() << con::log_level(con::error)
                      << "high_res_time_stamp::initialize(): "
                      << "error obtaining performance counter frequency" << std::endl
                      << " - system error message: " << std::endl << "    "
                      << error_msg
                      << std::endl;

        LocalFree(error_msg);

        return (false);
    }

    return (boost::numeric_cast<time_stamp>(frequency.QuadPart));
}

const time_stamp      pfc_frequency = get_pfc_frequency();

} // namespace 

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

        console.get() << con::log_level(con::error)
                      << "high_res_time_stamp::now(): "
                      << "error obtaining performance counter" << std::endl
                      << " - system error message: " << std::endl << "    "
                      << error_msg
                      << std::endl;

        LocalFree(error_msg);

        return (0);
    }

    return (boost::numeric_cast<time_stamp>(current_time_counter.QuadPart));
}

#elif    SCM_PLATFORM == SCM_PLATFORM_LINUX \
      || SCM_PLATFORM == SCM_PLATFORM_APPLE


#include <ctime>

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
    
    return timevalue(now.tv_sec - clock_process_initial.tv_sec, now.tv_nsec);

    return (  boost::numeric_cast<time_stamp>(current_time.tv_sec) * 1000000000
            + boost::numeric_cast<time_stamp>(current_time.tv_nsec));
}

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

} // namespace detail
} // namespace time
} // namespace scm

