
#include "accumulate_timer.h"

#include <CL/cl.hpp>

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

    cl_ulong start = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = _cl_event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong diff  = ((end > start) ? (end - start) : (~start + 1 + end));

    _accumulated_duration += time::nanosec(diff);
    ++_accumulation_count;
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
