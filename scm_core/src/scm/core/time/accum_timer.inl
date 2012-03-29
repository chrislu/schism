
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

namespace scm {
namespace time {

template<class timer_t>
accum_timer<timer_t>::accum_timer()
  : accum_timer_base_deprecated()
{
    _timer.start();
    _timer.stop();
}

template<class timer_t>
accum_timer<timer_t>::~accum_timer()
{
}

template<class timer_t>
void
accum_timer<timer_t>::start()
{
    _timer.start();
}

template<class timer_t>
void
accum_timer<timer_t>::stop()
{
    _timer.stop();

    _last_duration         = _timer.get_time();
    _accumulated_duration += _last_duration;
    ++_accumulation_count;
}

template<class timer_t>
void
accum_timer<timer_t>::collect()
{
}

template<class timer_t>
void
accum_timer<timer_t>::force_collect()
{
}

} // namespace time
} // namespace scm
