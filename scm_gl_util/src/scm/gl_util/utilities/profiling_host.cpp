
#include "profiling_host.h"

namespace scm {
namespace gl {
namespace util {

profiling_host::profiling_host()
{
}

profiling_host::~profiling_host()
{
    _cpu_timers.clear();
    _gl_timers.clear();
    _cl_timers.clear();
    _cu_timers.clear();
}

void
profiling_host::stop(const std::string& tname)
{
}

void
profiling_host::collect(const std::string& tname)
{
}

void
profiling_host::reset(const std::string& tname)
{
}

profiling_host::duration_type
profiling_host::accumulated_duration(const std::string& tname) const
{
    return duration_type();
}

unsigned
profiling_host::accumulation_count(const std::string& tname) const
{
    return 0u;
}

profiling_host::duration_type
profiling_host::average_duration(const std::string& tname) const
{
    return duration_type();
}

} // namespace util
} // namespace gl
} // namespace scm
