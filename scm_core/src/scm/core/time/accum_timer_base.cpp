
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "accum_timer_base.h"

namespace scm {
namespace time {

accum_timer_base::accum_timer_base()
  :  _last_time(0)
  , _accumulated_time(0)
  , _average_time(0)
  , _accumulation_count(0u)
  , _update_interval(0)
{
}

accum_timer_base::~accum_timer_base()
{
}

void
accum_timer_base::update(int interval)
{
    ++_update_interval;

    if (_update_interval >= interval) {
        _update_interval = 0;

        _average_time = (_accumulation_count > 0) ? _accumulated_time / _accumulation_count : 0;

        reset();
    }
}

void
accum_timer_base::reset()
{
    _last_time          = 0;
    _accumulated_time   = 0;
    _accumulation_count = 0u;
}

accum_timer_base::nanosec_type
accum_timer_base::last_time() const
{
    return _last_time;
}

accum_timer_base::nanosec_type
accum_timer_base::accumulated_time() const
{
    return _accumulated_time;
}

unsigned
accum_timer_base::accumulation_count() const
{
    return _accumulation_count;
}

accum_timer_base::nanosec_type
accum_timer_base::average_time() const
{
    return _average_time;
}

double
accum_timer_base::last_time(time_io::time_unit tu) const
{
    return time_io::to_time_unit(tu, last_time());
}

double
accum_timer_base::accumulated_time(time_io::time_unit tu) const
{
    return time_io::to_time_unit(tu, accumulated_time());
}

double
accum_timer_base::average_time(time_io::time_unit tu) const
{
    return time_io::to_time_unit(tu, average_time());
}

} // namespace time
} // namespace scm
