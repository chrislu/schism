
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "timer_base.h"

namespace scm {
namespace time {

timer_base::timer_base()
{
}

timer_base::~timer_base()
{
}

double
timer_base::elapsed(time_unit tu) const
{
    return to_time_unit(tu, elapsed());
}

double
timer_base::to_time_unit(time_unit tu, nanosec_type t)
{
    switch (tu) {
        case sec:  return static_cast<double>(t) * 0.000000001;
        case msec: return static_cast<double>(t) * 0.000001;
        case usec: return static_cast<double>(t) * 0.001;
        case nsec:
        default:   return static_cast<double>(t);
    }
}

std::string
timer_base::time_unit_string(time_unit tu)
{
    std::string r;

    switch (tu) {
        case sec:  r.assign("s");  break;
        case msec: r.assign("ms"); break;
        case usec: r.assign("us"); break;
        case nsec: r.assign("ns"); break;
        default:   r.assign("unknown");
    }

    return r;
}

} // namespace time
} // namespace scm
