
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <sstream>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

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
  : time::accum_timer_base()
  , _cu_event_finished(true)
  , _cu_event_srecorded(false)
  , _cu_event_erecorded(false)
  , _cu_event_stream(0)
  , _cpu_timer()
{
    reset();
    _detailed_average_time.cuda =
    _detailed_average_time.wall =
    _detailed_average_time.user =
    _detailed_average_time.system = 0;

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
        _cpu_timer.start();

        cudaError e = cudaSuccess;
        e = cudaEventRecord(_cu_event_start, cu_stream);
        _cu_event_srecorded = true;
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
        _cu_event_erecorded = true;
        assert(cudaSuccess == e);
        _cpu_timer.stop();
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
        if (_cu_event_erecorded && _cu_event_srecorded) {
            cudaError e = cudaStreamSynchronize(_cu_event_stream);
            assert(cudaSuccess == e);

            float cu_copy_time = 0.0f;
            e = cudaEventElapsedTime(&cu_copy_time, _cu_event_start, _cu_event_stop);
            assert(cudaSuccess == e);

            int64 ns = static_cast<int64>(static_cast<double>(cu_copy_time) * 1000.0 * 1000.0);
            time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

            _last_time         = static_cast<nanosec_type>(ns);
            _accumulated_time += _last_time;

            _detailed_last_time.cuda   = _last_time;
            _detailed_last_time.wall   = t.wall;
            _detailed_last_time.user   = t.user;
            _detailed_last_time.system = t.system;

            _detailed_accumulated_time.cuda   += _detailed_last_time.cuda;
            _detailed_accumulated_time.wall   += _detailed_last_time.wall;
            _detailed_accumulated_time.user   += _detailed_last_time.user;
            _detailed_accumulated_time.system += _detailed_last_time.system;

            ++_accumulation_count;
            _cu_event_finished  = true;
            _cu_event_srecorded = false;
            _cu_event_erecorded = false;
        }
    }
    else {
        err() << "accum_timer::collect() "
              << "error collecting cuda events (" << cudaGetErrorString(f) << ")." << log::end;
    }
}

void
accum_timer::force_collect()
{
    if (_cu_event_erecorded && _cu_event_srecorded) {

        cudaError e = cudaStreamSynchronize(_cu_event_stream);
        assert(cudaSuccess == e);

        float cu_copy_time = 0.0f;
        e = cudaEventElapsedTime(&cu_copy_time, _cu_event_start, _cu_event_stop);
        assert(cudaSuccess == e);

        int64 ns = static_cast<int64>(static_cast<double>(cu_copy_time) * 1000.0 * 1000.0);
        time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

        _last_time         = static_cast<nanosec_type>(ns);
        _accumulated_time += _last_time;

        _detailed_last_time.cuda   = _last_time;
        _detailed_last_time.wall   = t.wall;
        _detailed_last_time.user   = t.user;
        _detailed_last_time.system = t.system;

        _detailed_accumulated_time.cuda   += _detailed_last_time.cuda;
        _detailed_accumulated_time.wall   += _detailed_last_time.wall;
        _detailed_accumulated_time.user   += _detailed_last_time.user;
        _detailed_accumulated_time.system += _detailed_last_time.system;

        ++_accumulation_count;
        _cu_event_finished  = true;
        _cu_event_srecorded = false;
        _cu_event_erecorded = false;
    }
}

void
accum_timer::update(int interval)
{
    ++_update_interval;

    if (_update_interval >= interval) {
        _update_interval = 0;

        _average_time = (_accumulation_count > 0) ? _accumulated_time / _accumulation_count : 0;

        _detailed_average_time.cuda =
        _detailed_average_time.wall =
        _detailed_average_time.user =
        _detailed_average_time.system = 0;
        if (_accumulation_count > 0) {
            _detailed_average_time.cuda   = _detailed_accumulated_time.cuda   / _accumulation_count;
            _detailed_average_time.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
            _detailed_average_time.user   = _detailed_accumulated_time.user   / _accumulation_count;
            _detailed_average_time.system = _detailed_accumulated_time.system / _accumulation_count;
        }

        reset();
    }
}

void
accum_timer::reset()
{
    time::accum_timer_base::reset();
    _cu_event_finished  = true;
    _cu_event_srecorded = false;
    _cu_event_erecorded = false;

    _detailed_accumulated_time.cuda   = 
    _detailed_accumulated_time.wall   = 
    _detailed_accumulated_time.user   = 
    _detailed_accumulated_time.system = 0;
}

accum_timer::cu_times
accum_timer::detailed_last_time() const
{
    return _detailed_last_time;
}

accum_timer::cu_times
accum_timer::detailed_accumulated_time() const
{
    return _detailed_accumulated_time;
}

accum_timer::cu_times
accum_timer::detailed_average_time() const
{
    return _detailed_average_time;
}

void
accum_timer::report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cuda;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer::report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cuda;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, c, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

void
accum_timer::detailed_report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cuda;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "cuda " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer::detailed_report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type c  = detailed_average_time().cuda;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "cuda " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, c)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, c, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

} // namespace util
} // namespace cu
} // namespace scm
