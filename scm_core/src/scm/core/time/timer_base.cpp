
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
timer_base::elapsed(time_io::time_unit tu) const
{
    return time_io::to_time_unit(tu, elapsed());
}

double
time_io::to_time_unit(time_unit tu, nanosec_type t)
{
    switch (tu) {
        case sec:  return static_cast<double>(t) * 0.000000001;
        case msec: return static_cast<double>(t) * 0.000001;
        case usec: return static_cast<double>(t) * 0.001;
        case nsec:
        default:   return static_cast<double>(t);
    }
}

double
time_io::to_throughput_unit(throughput_unit tu, nanosec_type t, size_t d)
{
    double b = static_cast<double>(d);
    double s = to_time_unit(sec, t);

    switch (tu) {
        case Bps:                                       break;
        case KiBps: b = b / (1024.0);                   break;
        case MiBps: b = b / (1024.0 * 1024.0);          break;
        case GiBps: b = b / (1024.0 * 1024.0 * 1024.0); break;
    }

    return b / s;
}

std::string
time_io::time_unit_string(time_unit tu)
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

std::string
time_io::throughput_unit_string(throughput_unit tu)
{
    std::string r;

    switch (tu) {
        case Bps:   r.assign("B/s");   break;
        case KiBps: r.assign("KiB/s"); break;
        case MiBps: r.assign("MiB/s"); break;
        case GiBps: r.assign("GiB/s"); break;
        default:    r.assign("unknown");
    }

    return r;
}


} // namespace time
} // namespace scm
