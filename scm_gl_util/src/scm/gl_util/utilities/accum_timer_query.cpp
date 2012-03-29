
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer_query.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/query_objects/timer_query.h>

namespace scm {
namespace gl {

accum_timer_query::accum_timer_query(const render_device_ptr& device)
  : time::accum_timer_base_deprecated()
  , _timer_query_finished(true)
{
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

            _last_duration         = time::nanosec(diff);
            _accumulated_duration += _last_duration;
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

        _last_duration         = time::nanosec(diff);
        _accumulated_duration += _last_duration;
        ++_accumulation_count;
        _timer_query_finished = true;
    }
}

void
accum_timer_query::reset()
{
    time::accum_timer_base_deprecated::reset();

    _timer_query_finished   = true;
    _timer_context.reset();
}

} // namespace gl
} // namespace scm
