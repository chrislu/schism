
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer.h"

#include <CL/cl.hpp>
#include <scm/cl_core/opencl.h>

namespace scm {
namespace cl {
namespace util {

accum_timer::accum_timer()
  : time::accum_timer_base_deprecated()
  , _cl_event(new ::cl::Event())
  , _cl_event_finished(true)
{
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
        
        _last_duration         = time::nanosec(diff);
        _accumulated_duration += _last_duration;
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
        
    _last_duration         = time::nanosec(diff);
    _accumulated_duration += _last_duration;
    ++_accumulation_count;
    _cl_event_finished = true;
}

void
accum_timer::reset()
{
    time::accum_timer_base_deprecated::reset();
    _cl_event_finished    = false;
}

} // namespace util
} // namespace cl
} // namespace scm
