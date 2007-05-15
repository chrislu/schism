
#ifndef GET_TIME_H_INCLUDED
#define GET_TIME_H_INCLUDED

#include <scm_core/platform/platform.h>

namespace scm
{
    namespace core
    {
        namespace detail
        {
            // retrieves current system time value in milliseconds.
            // the resolution if this timer is at least microseconds.
            double __scm_export get_time();

        } // namespace detail
    } // namespace core
} // namespace scm

#endif //GET_TIME_H_INCLUDED
