
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_CUDA_DEVICE_H_INCLUDED
#define SCM_CL_CORE_CUDA_DEVICE_H_INCLUDED

#include <set>
#include <string>

#include <boost/noncopyable.hpp>

#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/cl_core/cuda/cuda_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cu {

class __scm_export(cl_core) cuda_device : boost::noncopyable
{
////// types //////////////////////////////////////////////////////////////////////////////////////
////// methods ////////////////////////////////////////////////////////////////////////////////////
public:
    cuda_device();
    virtual ~cuda_device();

    // device /////////////////////////////////////////////////////////////////////////////////////
public:
    int                             cu_device() const;

    cuda_command_stream_ptr         create_command_stream();

    virtual void                    print_device_informations(std::ostream& os) const;

protected:
    bool                            init_cuda();

////// attributes /////////////////////////////////////////////////////////////////////////////////
protected:
    // device /////////////////////////////////////////////////////////////////////////////////////
    struct mutex_impl;
    shared_ptr<mutex_impl>          _mutex_impl;

    // cuda ///////////////////////////////////////////////////////////////////////////////////////
    int                             _cuda_gl_device;

}; // class cuda_device

__scm_export(cl_core) std::ostream& operator<<(std::ostream& os, const cuda_device& ren_dev);

} // namespace cu
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // #define SCM_CL_CORE_CUDA_DEVICE_H_INCLUDED

