
#include "accumulate_timer_query.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/query_objects/timer_query.h>

namespace scm {
namespace gl {

accumulate_timer_query::accumulate_timer_query(const render_device_ptr& device)
  : _accumulated_duration(duration_type())
  , _accumulation_count(0u)
  , _timer_query_finished(true)
{
    _timer_query = device->create_timer_query();

    if (!_timer_query) {
        throw std::runtime_error("accumulate_timer_query::accumulate_timer_query(): error creating query object.");
    }
}

accumulate_timer_query::~accumulate_timer_query()
{
    _timer_query.reset();
}

void
accumulate_timer_query::start(const render_context_ptr& context)
{
    assert(_timer_query);
    if (_timer_query_finished) {
        context->begin_query(_timer_query);
    }
}

void
accumulate_timer_query::stop(const render_context_ptr& context)
{
    assert(_timer_query);
    if (_timer_query_finished) {
        context->end_query(_timer_query);
    }
}

void
accumulate_timer_query::collect(const render_context_ptr& context)
{
    assert(_timer_query);

    if (context->query_result_available(_timer_query)) {
        context->collect_query_results(_timer_query);

        _accumulated_duration += time::nanosec(_timer_query->result());
        ++_accumulation_count;
        _timer_query_finished = true;
    }
    else {
        _timer_query_finished = false;
    }
}

void
accumulate_timer_query::force_collect(const render_context_ptr& context)
{
    assert(_timer_query);

    context->collect_query_results(_timer_query);

    _accumulated_duration += time::nanosec(_timer_query->result());
    ++_accumulation_count;
    _timer_query_finished = true;
}

void
accumulate_timer_query::reset()
{
    _accumulated_duration   = duration_type();
    _accumulation_count     = 0u;
    _timer_query_finished   = true;
}

const accumulate_timer_query::duration_type&
accumulate_timer_query::accumulated_duration() const
{
    return _accumulated_duration;
}

unsigned
accumulate_timer_query::accumulation_count() const
{
    return _accumulation_count;
}

accumulate_timer_query::duration_type
accumulate_timer_query::average_duration() const
{
    if (_accumulation_count > 0) {
        return _accumulated_duration / _accumulation_count;
    }
    else {
        return duration_type();
    }
}

} // namespace gl
} // namespace scm
