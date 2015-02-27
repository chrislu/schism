
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer.h"

#include <ostream>
#include <iomanip>
#include <sstream>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

#include <scm/cl_core/opencl.h>
#include <scm/cl_core/opencl/CL/cl.hpp>

namespace scm {
namespace cl {
namespace util {

accum_timer::accum_timer()
  : time::accum_timer_base()
  , _cl_event(new ::cl::Event())
  , _cl_event_finished(true)
  , _cpu_timer()
{
    reset();
    _detailed_average_time.cl =
    _detailed_average_time.wall =
    _detailed_average_time.user =
    _detailed_average_time.system = 0;
}

accum_timer::~accum_timer()
{
    _cl_event.reset();
}

::cl::Event*const
accum_timer::event() const
{
    if (_cl_event_finished) {
        return _cl_event.get();
    }
    else {
        return 0;//nullptr;
    }
}

void
accum_timer::stop()
{
}

void
accum_timer::collect()
{
    assert(_cl_event);
    cl_int      cl_error00 = CL_SUCCESS;

    cl_ulong end   = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&cl_error00);

    if (CL_PROFILING_INFO_NOT_AVAILABLE == cl_error00) {
        _cl_event_finished = false;
        //gl::glerr() << "not finished";
        //gl::glerr() << log::error
        //            << "accum_timer::collect(): "
        //            << "unable retrieve timer data "
        //            << "(" << util::cl_error_string(cl_error00) << ", " << util::cl_error_string(cl_error01) << ")." << log::end;
    }
    else if (CL_SUCCESS == cl_error00)  {
        //gl::glerr() << "finished";
        cl_ulong start = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&cl_error00);
        cl_ulong diff  = ((end > start) ? (end - start) : (~start + 1 + end));
        
        time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

        _last_time         = static_cast<nanosec_type>(diff);
        _accumulated_time += _last_time;

        _detailed_last_time.cl     = _last_time;
        _detailed_last_time.wall   = t.wall;
        _detailed_last_time.user   = t.user;
        _detailed_last_time.system = t.system;

        _detailed_accumulated_time.cl     += _detailed_last_time.cl;
        _detailed_accumulated_time.wall   += _detailed_last_time.wall;
        _detailed_accumulated_time.user   += _detailed_last_time.user;
        _detailed_accumulated_time.system += _detailed_last_time.system;

        ++_accumulation_count;
        _cl_event_finished = true;
    }
}

void
accum_timer::force_collect()
{
    assert(_cl_event);
    cl_int      cl_error00 = CL_SUCCESS;

    cl_ulong end   = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&cl_error00);
    cl_ulong start = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&cl_error00);

    cl_ulong diff  = ((end > start) ? (end - start) : (~start + 1 + end));
        
    time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

    _last_time         = static_cast<nanosec_type>(diff);
    _accumulated_time += _last_time;

    _detailed_last_time.cl     = _last_time;
    _detailed_last_time.wall   = t.wall;
    _detailed_last_time.user   = t.user;
    _detailed_last_time.system = t.system;

    _detailed_accumulated_time.cl     += _detailed_last_time.cl;
    _detailed_accumulated_time.wall   += _detailed_last_time.wall;
    _detailed_accumulated_time.user   += _detailed_last_time.user;
    _detailed_accumulated_time.system += _detailed_last_time.system;

    ++_accumulation_count;
    _cl_event_finished = true;
}

void
accum_timer::update(int interval)
{
    ++_update_interval;

    if (_update_interval >= interval) {
        _update_interval = 0;

        _average_time = (_accumulation_count > 0) ? _accumulated_time / _accumulation_count : 0;

        _detailed_average_time.cl =
        _detailed_average_time.wall =
        _detailed_average_time.user =
        _detailed_average_time.system = 0;
        if (_accumulation_count > 0) {
            _detailed_average_time.cl     = _detailed_accumulated_time.cl     / _accumulation_count;
            _detailed_average_time.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
            _detailed_average_time.user   = _detailed_accumulated_time.user   / _accumulation_count;
            _detailed_average_time.system = _detailed_accumulated_time.system / _accumulation_count;
        }

        reset();
    }
}

void
accum_timer::reset()
{
    time::accum_timer_base::reset();
    _cl_event_finished    = false;

    _detailed_accumulated_time.cl     = 
    _detailed_accumulated_time.wall   = 
    _detailed_accumulated_time.user   = 
    _detailed_accumulated_time.system = 0;
}

accum_timer::cl_times
accum_timer::detailed_last_time() const
{
    return _detailed_last_time;
}

accum_timer::cl_times
accum_timer::detailed_accumulated_time() const
{
    return _detailed_accumulated_time;
}

accum_timer::cl_times
accum_timer::detailed_average_time() const
{
    return _detailed_average_time;
}

void
accum_timer::report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cl;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer::report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cl;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, c, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

void
accum_timer::detailed_report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "cl   " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer::detailed_report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "cl   " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, c, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

} // namespace util
} // namespace cl
} // namespace scm
