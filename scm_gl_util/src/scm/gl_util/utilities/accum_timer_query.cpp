
#include "accum_timer_query.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/query_objects/timer_query.h>

namespace scm {
namespace gl {

accum_timer_query::accum_timer_query(const render_device_ptr& device)
  : time::accum_timer_base()
  , _timer_query_finished(true)
{
    _timer_query = device->create_timer_query();

    if (!_timer_query) {
        throw std::runtime_error("accum_timer_query::accum_timer_query(): error creating query object.");
    }
}

accum_timer_query::~accum_timer_query()
{
    _timer_query.reset();
}

void
accum_timer_query::start(const render_context_ptr& context)
{
    assert(_timer_query);
    if (_timer_query_finished) {
        context->begin_query(_timer_query);
        _timer_context = context;
    }
}

void
accum_timer_query::stop()
{
    assert(_timer_query);
    if (_timer_query_finished && _timer_context) {
        _timer_context->end_query(_timer_query);
    }
}

void
accum_timer_query::collect()
{
    assert(_timer_query);

    if (_timer_context) {
        if (_timer_context->query_result_available(_timer_query)) {
            _timer_context->collect_query_results(_timer_query);

            _last_duration         = time::nanosec(_timer_query->result());
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
    assert(_timer_query);

    if (_timer_context) {
        _timer_context->collect_query_results(_timer_query);

        _last_duration         = time::nanosec(_timer_query->result());
        _accumulated_duration += _last_duration;
        ++_accumulation_count;
        _timer_query_finished = true;
    }
}

void
accum_timer_query::reset()
{
    time::accum_timer_base::reset();

    _timer_query_finished   = true;
    _timer_context.reset();
}

} // namespace gl
} // namespace scm
