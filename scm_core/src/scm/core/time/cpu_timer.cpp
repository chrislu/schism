
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "cpu_timer.h"

#include <ostream>
#include <iomanip>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

namespace scm {
namespace time {

cpu_timer::cpu_timer()
  : _timer()
{
}

cpu_timer::~cpu_timer()
{
}

void
cpu_timer::start()
{
    _timer.start();
}

void
cpu_timer::stop()
{
    _timer.stop();
}

bool
cpu_timer::result_available() const
{
    return true;
}

void
cpu_timer::collect() const
{
}

void
cpu_timer::force_collect() const
{
}

cpu_timer::nanosec_type
cpu_timer::elapsed() const
{
    return _timer.elapsed().wall;
}

cpu_timer::cpu_times
cpu_timer::detailed_elapsed() const
{
    return _timer.elapsed();
}

void
cpu_timer::report(std::ostream& os, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_elapsed().wall;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
cpu_timer::report(std::ostream& os, size_t dsize, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        nanosec_type w  = detailed_elapsed().wall;


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
cpu_timer::detailed_report(std::ostream& os, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_elapsed().wall;
        nanosec_type u  = detailed_elapsed().user;
        nanosec_type s  = detailed_elapsed().system;
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
cpu_timer::detailed_report(std::ostream& os, size_t dsize, time_io unit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type w  = detailed_elapsed().wall;
        nanosec_type u  = detailed_elapsed().user;
        nanosec_type s  = detailed_elapsed().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "user " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, u)  << time_io::time_unit_string(unit._t_unit) << "+ "
           << "sys "  << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, s)  << time_io::time_unit_string(unit._t_unit) << "= "
           <<            std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, us) << time_io::time_unit_string(unit._t_unit)
           << " (" << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

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
