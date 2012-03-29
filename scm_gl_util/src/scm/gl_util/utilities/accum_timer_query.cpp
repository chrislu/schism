
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer_query.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <sstream>

#include <boost/chrono.hpp>
#include <boost/io/ios_state.hpp>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/query_objects/timer_query.h>

namespace scm {
namespace gl {

accum_timer_query::accum_timer_query(const render_device_ptr& device)
  : time::accum_timer_base()
  , _timer_query_finished(true)
  , _cpu_timer()
{
    reset();
    _detailed_average_time.gl =
    _detailed_average_time.wall =
    _detailed_average_time.user =
    _detailed_average_time.system = 0;

    _timer_query_begin = device->create_timer_query();
    _timer_query_end = device->create_timer_query();

    if (   !_timer_query_begin
        || !_timer_query_end) {
        throw std::runtime_error("accum_timer_query::accum_timer_query(): error creating query object.");
    }
}

accum_timer_query::~accum_timer_query()
{
    _timer_query_begin.reset();
    _timer_query_end.reset();
}

void
accum_timer_query::start(const render_context_ptr& context)
{
    assert(_timer_query_begin);
    assert(_timer_query_end);

    if (_timer_query_finished) {
        _cpu_timer.start();
        context->query_time_stamp(_timer_query_begin);
        _timer_context = context;
    }
}

void
accum_timer_query::stop()
{
    assert(_timer_query_begin);
    assert(_timer_query_end);

    if (_timer_query_finished && _timer_context) {
        _timer_context->query_time_stamp(_timer_query_end);
        _cpu_timer.stop();
    }
}

void
accum_timer_query::collect()
{
    assert(_timer_query_begin);
    assert(_timer_query_end);

    if (_timer_context) {
        if (_timer_context->query_result_available(_timer_query_end)) {
            _timer_context->collect_query_results(_timer_query_begin);
            _timer_context->collect_query_results(_timer_query_end);

            scm::uint64 start = _timer_query_begin->result();
            scm::uint64 end   = _timer_query_end->result();
            scm::uint64 diff  = ((end > start) ? (end - start) : (~start + 1 + end));

            time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

            _last_time         = static_cast<nanosec_type>(diff);
            _accumulated_time += _last_time;

            _detailed_last_time.gl     = _last_time;
            _detailed_last_time.wall   = t.wall;
            _detailed_last_time.user   = t.user;
            _detailed_last_time.system = t.system;

            _detailed_accumulated_time.gl     += _detailed_last_time.gl;
            _detailed_accumulated_time.wall   += _detailed_last_time.wall;
            _detailed_accumulated_time.user   += _detailed_last_time.user;
            _detailed_accumulated_time.system += _detailed_last_time.system;

            ++_accumulation_count;
            _timer_query_finished = true;
        }
        else {
            _timer_query_finished = false;
        }
    }
}

void
accum_timer_query::force_collect()
{
    assert(_timer_query_begin);
    assert(_timer_query_end);

    if (_timer_context) {
        _timer_context->collect_query_results(_timer_query_begin);
        _timer_context->collect_query_results(_timer_query_end);

        scm::uint64 start = _timer_query_begin->result();
        scm::uint64 end   = _timer_query_end->result();
        scm::uint64 diff  = ((end > start) ? (end - start) : (~start + 1 + end));

        time::cpu_timer::cpu_times t = _cpu_timer.detailed_elapsed();

        _last_time         = static_cast<nanosec_type>(diff);
        _accumulated_time += _last_time;

        _detailed_last_time.gl     = _last_time;
        _detailed_last_time.wall   = t.wall;
        _detailed_last_time.user   = t.user;
        _detailed_last_time.system = t.system;

        _detailed_accumulated_time.gl     += _detailed_last_time.gl;
        _detailed_accumulated_time.wall   += _detailed_last_time.wall;
        _detailed_accumulated_time.user   += _detailed_last_time.user;
        _detailed_accumulated_time.system += _detailed_last_time.system;

        ++_accumulation_count;
        _timer_query_finished = true;
    }
}

void
accum_timer_query::update(int interval)
{
    ++_update_interval;

    if (_update_interval >= interval) {
        _update_interval = 0;

        _average_time = (_accumulation_count > 0) ? _accumulated_time / _accumulation_count : 0;

        _detailed_average_time.gl =
        _detailed_average_time.wall =
        _detailed_average_time.user =
        _detailed_average_time.system = 0;
        if (_accumulation_count > 0) {
            _detailed_average_time.gl     = _detailed_accumulated_time.gl     / _accumulation_count;
            _detailed_average_time.wall   = _detailed_accumulated_time.wall   / _accumulation_count;
            _detailed_average_time.user   = _detailed_accumulated_time.user   / _accumulation_count;
            _detailed_average_time.system = _detailed_accumulated_time.system / _accumulation_count;
        }

        reset();
    }
}

void
accum_timer_query::reset()
{
    time::accum_timer_base::reset();

    _timer_query_finished   = true;
    _timer_context.reset();

    _detailed_accumulated_time.gl     = 
    _detailed_accumulated_time.wall   = 
    _detailed_accumulated_time.user   = 
    _detailed_accumulated_time.system = 0;
}

accum_timer_query::gl_times
accum_timer_query::detailed_last_time() const
{
    return _detailed_last_time;
}

accum_timer_query::gl_times
accum_timer_query::detailed_accumulated_time() const
{
    return _detailed_accumulated_time;
}

accum_timer_query::gl_times
accum_timer_query::detailed_average_time() const
{
    return _detailed_average_time;
}

void
accum_timer_query::report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type g  = detailed_average_time().gl;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, g)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer_query::report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type g  = detailed_average_time().gl;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << time_io::to_time_unit(unit._t_unit, g)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, g, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}

void
accum_timer_query::detailed_report(std::ostream& os, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        nanosec_type g  = detailed_average_time().gl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "gl   " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, g)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);
    }
}

void
accum_timer_query::detailed_report(std::ostream& os, size_t dsize, time::time_io unit) const
{
    using namespace scm::time;

    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        nanosec_type g  = detailed_average_time().gl;
        nanosec_type w  = detailed_average_time().wall;
        nanosec_type u  = detailed_average_time().user;
        nanosec_type s  = detailed_average_time().system;
        nanosec_type us = u + s;

        os << std::fixed << std::setprecision(unit._t_dec_places)
           << "gl   " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, g)  << time_io::time_unit_string(unit._t_unit) << ", "
           << "wall " << std::setw(unit._t_dec_places + 3) << std::right << time_io::to_time_unit(unit._t_unit, w)  << time_io::time_unit_string(unit._t_unit);

        if (0 < dsize) {
            os << ", " << std::fixed << std::setprecision(unit._tp_dec_places)
               << std::setw(unit._tp_dec_places + 5) << std::right
               << time_io::to_throughput_unit(unit._tp_unit, g, dsize)
               << time_io::throughput_unit_string(unit._tp_unit);
        }
    }
}


} // namespace gl
} // namespace scm
