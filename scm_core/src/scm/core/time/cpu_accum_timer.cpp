
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "cpu_accum_timer.h"

#include <ostream>
#include <iomanip>
#include <sstream>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

namespace scm {
namespace time {

cpu_accum_timer::cpu_accum_timer()
  : _cpu_timer()
{
    reset();
    _cpu_timer.start();
    _cpu_timer.stop();
}

cpu_accum_timer::~cpu_accum_timer()
{
}

void
cpu_accum_timer::start()
{
    _cpu_timer.start();
}

void
cpu_accum_timer::stop()
{
    _cpu_timer.stop();

    cpu_times t = _cpu_timer.detailed_elapsed();

    _last_time         = t.wall;
    _accumulated_time += _last_time;

    _detailed_last_time.wall   = t.wall;
    _detailed_last_time.user   = t.user;
    _detailed_last_time.system = t.system;

    _detailed_accumulated_time.wall   += _detailed_last_time.wall;
    _detailed_accumulated_time.user   += _detailed_last_time.user;
    _detailed_accumulated_time.system += _detailed_last_time.system;

    ++_accumulation_count;
}

void
cpu_accum_timer::collect()
{
}

void
cpu_accum_timer::force_collect()
{
}

void
cpu_accum_timer::reset()
{
    accum_timer_base::reset();

    _detailed_accumulated_time.wall = 
    _detailed_accumulated_time.user = 
    _detailed_accumulated_time.system = 0;
}

cpu_accum_timer::cpu_times
cpu_accum_timer::detailed_last_time() const
{
    return _detailed_last_time;
}

cpu_accum_timer::cpu_times
cpu_accum_timer::detailed_accumulated_time() const
{
    return _detailed_accumulated_time;
}

cpu_accum_timer::cpu_times
cpu_accum_timer::detailed_average_time() const
{
    cpu_times avg;
    avg.wall = avg.user = avg.system = 0;
    if (_accumulation_count > 0) {
        avg.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
        avg.user   = _detailed_accumulated_time.user   / _accumulation_count;
        avg.system = _detailed_accumulated_time.system / _accumulation_count;
    }

    return avg;
}

void
cpu_accum_timer::report(std::ostream&               os,
                        timer_base::time_unit       tunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_average_time().wall;

        os << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit);
    }
}

void
cpu_accum_timer::report(std::ostream&               os,
                        size_t                      dsize,
                        timer_base::time_unit       tunit,
                        timer_base::throughput_unit tpunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_average_time().wall;

        os << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit);

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, w, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

void
cpu_accum_timer::detailed_report(std::ostream&               os,
                                 timer_base::time_unit       tunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        std::stringstream percent;
        percent << "(" << std::fixed << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

        os << "wall:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit) << ", "
           << "user:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, u)  << timer_base::time_unit_string(tunit) << ", "
           << "sys:"  << std::setw(6) << std::right << timer_base::to_time_unit(tunit, s)  << timer_base::time_unit_string(tunit) << ", "
           <<             std::setw(6) << std::right << timer_base::to_time_unit(tunit, us) << timer_base::time_unit_string(tunit)
           << std::setw(9) << std::right << percent.str();;
    }
}

void
cpu_accum_timer::detailed_report(std::ostream&               os,
                                 size_t                      dsize,
                                 timer_base::time_unit       tunit,
                                 timer_base::throughput_unit tpunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        std::stringstream percent;
        percent << "(" << std::fixed << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

        os << "wall:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit) << ", "
           << "user:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, u)  << timer_base::time_unit_string(tunit) << ", "
           << "sys:"  << std::setw(6) << std::right << timer_base::to_time_unit(tunit, s)  << timer_base::time_unit_string(tunit) << ", "
           <<             std::setw(6) << std::right << timer_base::to_time_unit(tunit, us) << timer_base::time_unit_string(tunit)
           << std::setw(9) << std::right << percent.str();;

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, w, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

} // namespace time
} // namespace scm
