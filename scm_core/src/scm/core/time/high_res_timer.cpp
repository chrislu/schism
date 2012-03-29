
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "high_res_timer.h"

#include <scm/core/utilities/static_global.h>
#include <scm/core/time/time_system.h>
#include <scm/core/time/detail/highres_time_stamp.h>

namespace  {

SCM_STATIC_GLOBAL(scm::time::detail::high_res_time_stamp, global_high_res_time_stamp)

} // namespace 


namespace scm {
namespace time {

high_res_timer::high_res_timer(resolution_type res_type)
  : _start(0),
    timer_interface(res_type)
{
}

high_res_timer::~high_res_timer()
{

}

void high_res_timer::start()
{
    _start = now();
}

void high_res_timer::stop()
{
    _duration = elapsed_duration(_start, now());
}

void high_res_timer::intermediate_stop()
{
    _duration = elapsed_duration(_start, now());
}

void high_res_timer::collect_result() const
{
}

time_stamp
high_res_timer::now() const
{
    return (global_high_res_time_stamp().now());
}

high_res_timer::duration_type
high_res_timer::elapsed_duration(time_stamp start,
                                 time_stamp end) const
{
    // look into wrap arounds
    time_stamp  diff((end > start) ? (end - start) : (~start + 1 + end));

    double      dur =   static_cast<double>(diff)
                      / static_cast<double>(global_high_res_time_stamp().ticks_per_second());
    
    switch (resolution()) {
        case nano_seconds:      return (nanosec(static_cast<time_stamp>(dur * 1e9)));break;
        case micro_seconds:     return (microsec(static_cast<time_stamp>(dur * 1e6)));break;
        case milli_seconds:     return (millisec(static_cast<time_stamp>(dur * 1e3)));break;
    }
    return (microsec(0));
}

} // namespace time
} // namespace scm
