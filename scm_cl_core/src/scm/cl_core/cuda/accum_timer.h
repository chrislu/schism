
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_CUDA_ACCUMU_TIMER_H_INCLUDED
#define SCM_CL_CORE_CUDA_ACCUMU_TIMER_H_INCLUDED

#include <scm/core/memory.h>
#include <scm/core/time/accum_timer.h>

#include <scm/cl_core/cuda/cuda_fwd.h>

#include <cuda_runtime_api.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cu {
namespace util {

class __scm_export(cl_core) accum_timer : public time::accum_timer_base
{
public:
    accum_timer();
    virtual ~accum_timer();

    void                    start(cudaStream_t cu_stream = 0);
    void                    stop();
    void                    collect();
    void                    force_collect();
    void                    reset();

protected:
    bool                    _cu_event_finished;
    cudaEvent_t             _cu_event_start;
    cudaEvent_t             _cu_event_stop;
    cudaStream_t            _cu_event_stream;

}; // class accum_timer

} // namespace util
} // namespace cu
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CL_CORE_CUDA_ACCUMU_TIMER_H_INCLUDED
