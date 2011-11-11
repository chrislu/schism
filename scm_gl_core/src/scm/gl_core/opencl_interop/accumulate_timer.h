
#ifndef SCM_GL_CORE_ACCUMULATE_TIMER_H_INCLUDED
#define SCM_GL_CORE_ACCUMULATE_TIMER_H_INCLUDED

#include <scm/core/memory.h>
#include <scm/core/time/time_types.h>

#include <scm/gl_core/opencl_interop/opencl_interop_fwd.h>

#include <CL/cl_fwd.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cl {
namespace util {

class __scm_export(gl_core) accumulate_timer
{
public:
    typedef time::time_duration      duration_type;

public:
    accumulate_timer();
    /*virtual*/ ~accumulate_timer();

    ::cl::Event*const       event() const;

    void                    collect();
    void                    reset();

    const duration_type&    accumulated_duration() const;
    unsigned                accumulation_count() const;

    duration_type           average_duration() const;

protected:
    duration_type           _accumulated_duration;
    unsigned                _accumulation_count;

    event_ptr               _cl_event;

}; // class accumulate_timer

} // namespace util
} // namespace cl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_ACCUMULATE_TIMER_H_INCLUDED
