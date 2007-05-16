
#include "get_time.h"

#include <cassert>

namespace scm
{
    namespace core
    {
        namespace detail
        {
            #if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
                #include <windows.h>

                namespace
                {
                    struct performance_counter
                    {
                        performance_counter()
                        {
                            LARGE_INTEGER       frequency;
                            QueryPerformanceFrequency(&frequency);

                            assert(frequency.QuadPart != 0);

                            _frequency              = static_cast<double>(frequency.QuadPart);
                            _current_time.QuadPart  = 0;
                        }
                        double              _frequency;
                        LARGE_INTEGER       _current_time;
                    }; // struct performance_counter

                    static performance_counter _performance_counter;
                } // namespace 

                double get_time()
                {
                    QueryPerformanceCounter(&_performance_counter._current_time);

                    assert(_performance_counter._current_time.QuadPart != 0);
                    
                    return (  (static_cast<double>(_performance_counter._current_time.QuadPart)
                            / _performance_counter._frequency)
                            * 1000.0);
                }

            #elif    SCM_PLATFORM == SCM_PLATFORM_LINUX \
                  || SCM_PLATFORM == SCM_PLATFORM_APPLE

                #include <sys/time.h>
                #include <time.h>

                namespace
                {
                    static timeval _time_values;
                } // namespace

                inline double get_time()
                {
                    gettimeofday(&_time_values, 0);

                    return (   1000.0 * static_cast<double>(_time_values.tv_sec)
                             + 0.001  * static_cast<double>(_time_values.tv_usec));
                }

            #endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

        } // namespace detail
    } // namespace core
} // namespace scm

