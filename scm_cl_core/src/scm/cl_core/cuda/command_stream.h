
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_CUDA_COMMAND_STREAM_H_INCLUDED
#define SCM_CL_CORE_CUDA_COMMAND_STREAM_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <cuda_runtime.h>

#include <scm/core/memory.h>

#include <scm/cl_core/cuda/cuda_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cu {

class __scm_export(cl_core) cuda_command_stream : boost::noncopyable
{
public:
    ~cuda_command_stream();

public:
    const cudaStream_t              stream() const;

protected:
    cuda_command_stream(cuda_device& cudev);

    cudaStream_t                    _stream;

private:
    friend class cuda_device;

}; // class cuda_command_stream

} // namespace cu
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // #define SCM_CL_CORE_CUDA_COMMAND_STREAM_H_INCLUDED

