
#ifndef SCM_CL_CORE_OPENCL_ACCUMU_TIMER_H_INCLUDED
#define SCM_CL_CORE_OPENCL_ACCUMU_TIMER_H_INCLUDED

#include <scm/core/memory.h>
#include <scm/core/time/accum_timer.h>

#include <scm/cl_core/opencl/opencl_fwd.h>

#include <CL/cl_fwd.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cl {
namespace util {

class __scm_export(cl_core) accum_timer : public time::accum_timer_base
{
public:
    accum_timer();
    virtual ~accum_timer();

    ::cl::Event*const       event() const;

    void                    stop();
    void                    collect();
    void                    force_collect();
    void                    reset();

protected:
    event_ptr               _cl_event;
    bool                    _cl_event_finished;

}; // class accum_timer

} // namespace util
} // namespace cl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CL_CORE_OPENCL_ACCUMU_TIMER_H_INCLUDED
