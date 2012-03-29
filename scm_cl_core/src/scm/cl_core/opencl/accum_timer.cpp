
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer.h"

#include <ostream>
#include <iomanip>
#include <sstream>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

#include <CL/cl.hpp>
#include <scm/cl_core/opencl.h>

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
    cl_times avg;
    avg.cl = avg.wall = avg.user = avg.system = 0;
    if (_accumulation_count > 0) {
        avg.cl     = _detailed_accumulated_time.cl     / _accumulation_count;
        avg.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
        avg.user   = _detailed_accumulated_time.user   / _accumulation_count;
        avg.system = _detailed_accumulated_time.system / _accumulation_count;
    }

    return avg;
}

void
accum_timer::report(std::ostream&                     os,
                    time::timer_base::time_unit       tunit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        nanosec_type c  = detailed_average_time().cl;

        os << timer_base::to_time_unit(tunit, c) << timer_base::time_unit_string(tunit);
    }
}

void
accum_timer::report(std::ostream&                     os,
                    size_t                            dsize,
                    time::timer_base::time_unit       tunit,
                    time::timer_base::throughput_unit tpunit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        nanosec_type c  = detailed_average_time().cl;

        os << timer_base::to_time_unit(tunit, c) << timer_base::time_unit_string(tunit);

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, c, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

void
accum_timer::detailed_report(std::ostream&                     os,
                             time::timer_base::time_unit       tunit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        nanosec_type c  = detailed_average_time().cl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << "cl:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, c)  << timer_base::time_unit_string(tunit) << ", "
           << "wall:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit);
    }
}

void
accum_timer::detailed_report(std::ostream&                     os,
                             size_t                            dsize,
                             time::timer_base::time_unit       tunit,
                             time::timer_base::throughput_unit tpunit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        nanosec_type c  = detailed_average_time().cl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << "cl:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, c)  << timer_base::time_unit_string(tunit) << ", "
           << "wall:" << std::setw(6) << std::right << timer_base::to_time_unit(tunit, w)  << timer_base::time_unit_string(tunit);

        if (0 < dsize) {
            os << ", "
                << std::setw(9) << std::right << timer_base::to_throughput_unit(tpunit, c, dsize)
                                              << timer_base::throughput_unit_string(tpunit);
        }
    }
}

} // namespace util
} // namespace cl
} // namespace scm
