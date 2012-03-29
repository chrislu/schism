
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
cpu_timer::report(std::ostream&   os,
                  time_unit       tunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_elapsed().wall;

        os << to_time_unit(tunit, w)  << time_unit_string(tunit);
    }
}

void
cpu_timer::report(std::ostream&   os,
                  size_t          dsize,
                  time_unit       tunit,
                  throughput_unit tpunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_elapsed().wall;

        os << to_time_unit(tunit, w)  << time_unit_string(tunit);

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, w, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

void
cpu_timer::detailed_report(std::ostream&   os,
                           time_unit       tunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_elapsed().wall;
        nanosec_type u  = detailed_elapsed().user;
        nanosec_type s  = detailed_elapsed().system;
        nanosec_type us = u + s;

        os << "wall: " << std::setw(6) << std::right << to_time_unit(tunit, w)  << time_unit_string(tunit) << ", "
           << "user: " << std::setw(6) << std::right << to_time_unit(tunit, u)  << time_unit_string(tunit) << "+ "
           << "sys: "  << std::setw(6) << std::right << to_time_unit(tunit, s)  << time_unit_string(tunit) << "= "
           <<             std::setw(6) << std::right << to_time_unit(tunit, us) << time_unit_string(tunit)
           << " (" << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";
    }
}

void
cpu_timer::detailed_report(std::ostream&   os,
                           size_t          dsize,
                           time_unit       tunit,
                           throughput_unit tpunit) const
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(2);

        nanosec_type w  = detailed_elapsed().wall;
        nanosec_type u  = detailed_elapsed().user;
        nanosec_type s  = detailed_elapsed().system;
        nanosec_type us = u + s;

        os << "wall: " << std::setw(6) << std::right << to_time_unit(tunit, w)  << time_unit_string(tunit) << ", "
           << "user: " << std::setw(6) << std::right << to_time_unit(tunit, u)  << time_unit_string(tunit) << "+ "
           << "sys: "  << std::setw(6) << std::right << to_time_unit(tunit, s)  << time_unit_string(tunit) << "= "
           <<             std::setw(6) << std::right << to_time_unit(tunit, us) << time_unit_string(tunit)
           << " (" << std::setprecision(1) << (static_cast<double>(us)/static_cast<double>(w)) * 100.0 << "%)";

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, w, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

} // namespace time
} // namespace scm
