
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

    _detailed_average_time.wall = 
    _detailed_average_time.user = 
    _detailed_average_time.system = 0;

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
cpu_accum_timer::update(int interval)
{
    ++_update_interval;

    if (_update_interval >= interval) {
        _update_interval = 0;

        _average_time = (_accumulation_count > 0) ? _accumulated_time / _accumulation_count : 0;

        _detailed_average_time.wall =
        _detailed_average_time.user =
        _detailed_average_time.system = 0;
        if (_accumulation_count > 0) {
            _detailed_average_time.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
            _detailed_average_time.user   = _detailed_accumulated_time.user   / _accumulation_count;
            _detailed_average_time.system = _detailed_accumulated_time.system / _accumulation_count;
        }

        reset();
    }
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
    return _detailed_average_time;
}

void
cpu_accum_timer::report(std::ostream& os, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_average_time().wall;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
cpu_accum_timer::report(std::ostream& os, size_t dsize, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_average_time().wall;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, w, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

void
cpu_accum_timer::detailed_report(std::ostream& os, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        std::stringstream percent;
        percent << "(" << std::fixed << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "user " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, u)  << time_io::time_unit_string(unit._t_unit) << "+ "
           << "sys "  << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, s)  << time_io::time_unit_string(unit._t_unit) << "= "
           <<            std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, us) << time_io::time_unit_string(unit._t_unit)
           << std::setw(9) << std::right << percent.str();
    }
}

void
cpu_accum_timer::detailed_report(std::ostream& os, size_t dsize, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        std::stringstream percent;
        percent << "(" << std::fixed << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "user " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, u)  << time_io::time_unit_string(unit._t_unit) << "+ "
           << "sys "  << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, s)  << time_io::time_unit_string(unit._t_unit) << "= "
           <<            std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, us) << time_io::time_unit_string(unit._t_unit)
           << std::setw(9) << std::right << percent.str();

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, w, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

} // namespace time
} // namespace scm
