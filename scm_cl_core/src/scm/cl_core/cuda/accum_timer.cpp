
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <scm/core/platform/windows.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <scm/log.h>
#include <scm/core/numeric_types.h>

namespace scm {
namespace cu {
namespace util {

accum_timer::accum_timer()
  : time::accum_timer_base_deprecated()
  , _cu_event_finished(true)
  , _cu_event_stream(0)
{
    cudaError b = cudaEventCreate(&_cu_event_start);
    cudaError e = cudaEventCreate(&_cu_event_stop);

    if (   cudaSuccess != b
        || cudaSuccess != e) {
        throw std::runtime_error("cu::accum_timer::accum_timer(): error creating cuda events.");
    }
}

accum_timer::~accum_timer()
{
    cudaEventDestroy(_cu_event_start);
    cudaEventDestroy(_cu_event_stop);
}

void
accum_timer::start(cudaStream_t cu_stream)
{
    if (_cu_event_finished) {
        cudaError e = cudaSuccess;
        e = cudaEventRecord(_cu_event_start, cu_stream);
        assert(cudaSuccess == e);
        _cu_event_stream = cu_stream;
    }
}

void
accum_timer::stop()
{
    if (_cu_event_finished) {
        cudaError e = cudaSuccess;
        e = cudaEventRecord(_cu_event_stop, _cu_event_stream);
        assert(cudaSuccess == e);
    }
}

void
accum_timer::collect()
{
    cudaError f = cudaEventQuery(_cu_event_stop);

    if (cudaErrorNotReady == f) {
        _cu_event_finished = false;
    }
    else if (cudaSuccess == f) {
        cudaError e = cudaStreamSynchronize(_cu_event_stream);
        assert(cudaSuccess == e);

        float cu_copy_time = 0.0f;
        e = cudaEventElapsedTime(&cu_copy_time, _cu_event_start, _cu_event_stop);
        assert(cudaSuccess == e);

        int64 ns = static_cast<int64>(static_cast<double>(cu_copy_time) * 1000.0 * 1000.0);
        _last_duration         = time::nanosec(ns);
        _accumulated_duration += _last_duration;
        ++_accumulation_count;
        _cu_event_finished = true;
    }
    else {
        err() << "accum_timer::collect() "
              << "error collecting cuda events (" << cudaGetErrorString(f) << ")." << log::end;
    }
}

void
accum_timer::force_collect()
{
    cudaError e = cudaStreamSynchronize(_cu_event_stream);
    assert(cudaSuccess == e);

    float cu_copy_time = 0.0f;
    e = cudaEventElapsedTime(&cu_copy_time, _cu_event_start, _cu_event_stop);
    assert(cudaSuccess == e);

    int64 ns = static_cast<int64>(static_cast<double>(cu_copy_time) * 1000.0 * 1000.0);
    _accumulated_duration += time::nanosec(ns);
    ++_accumulation_count;
    _cu_event_finished = true;
}

void
accum_timer::reset()
{
    time::accum_timer_base_deprecated::reset();
    _cu_event_finished    = false;
}

} // namespace util
} // namespace cu
} // namespace scm
