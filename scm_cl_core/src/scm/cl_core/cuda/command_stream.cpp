
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "command_stream.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <scm/log.h>

#include <scm/cl_core/config.h>
#include <scm/cl_core/cuda.h>

namespace scm {
namespace cu {

cuda_command_stream::cuda_command_stream(cuda_device& cudev)
  : _stream(0)
{
    cudaError cu_err = cudaSuccess;
    cu_err = cudaStreamCreate(&_stream);
    if (cudaSuccess != cu_err) {
        std::ostringstream s;
        s << "cuda_command_stream::cuda_command_stream() "
          << "error creating cuda stream (" << cudaGetErrorString(cu_err) << ").";
        throw std::runtime_error(s.str());
    }
}

cuda_command_stream::~cuda_command_stream()
{
    cudaStreamDestroy(_stream);
}

const cudaStream_t
cuda_command_stream::stream() const
{
    return _stream;
}

} // namespace cu
} // namespace scm
