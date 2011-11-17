
#include "accumulate_timer.h"

#include <CL/cl.hpp>
#include <scm/gl_core/opencl_interop.h>

#include <scm/gl_core/log.h>

namespace scm {
namespace cl {
namespace util {

accumulate_timer::accumulate_timer()
  : _cl_event(new ::cl::Event())
{
}

accumulate_timer::~accumulate_timer()
{
    _cl_event.reset();
}

::cl::Event*const
accumulate_timer::event() const
{
    return _cl_event.get();
}

void
accumulate_timer::collect()
{
    assert(_cl_event);
    cl_int      cl_error00 = CL_SUCCESS;
    cl_int      cl_error01 = CL_SUCCESS;

    cl_ulong start = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&cl_error00);
    cl_ulong end   = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&cl_error01);

    if (   CL_SUCCESS != cl_error00
        || CL_SUCCESS != cl_error01) {
        gl::glerr() << log::error
                    << "accumulate_timer::collect(): "
                    << "unable retrieve timer data "
                    << "(" << util::cl_error_string(cl_error00) << ", " << util::cl_error_string(cl_error01) << ")." << log::end;
    }
    else {
        cl_ulong diff  = ((end > start) ? (end - start) : (~start + 1 + end));
        _accumulated_duration += time::nanosec(diff);
        ++_accumulation_count;
    }
}

void
accumulate_timer::reset()
{
    _accumulated_duration   = duration_type();
    _accumulation_count     = 0u;
}

const accumulate_timer::duration_type&
accumulate_timer::accumulated_duration() const
{
    return _accumulated_duration;
}

unsigned
accumulate_timer::accumulation_count() const
{
    return _accumulation_count;
}

accumulate_timer::duration_type
accumulate_timer::average_duration() const
{
    if (_accumulation_count > 0) {
        return _accumulated_duration / _accumulation_count;
    }
    else {
        return duration_type();
    }
}

} // namespace util
} // namespace cl
} // namespace scm
