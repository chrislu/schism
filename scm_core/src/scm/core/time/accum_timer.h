
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED
#define SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED

#include <scm/core/memory.h>
#include <scm/core/time/time_types.h>

namespace scm {
namespace time {

class accum_timer_base
{
public:
    typedef time::time_duration     duration_type;

public:
    accum_timer_base() 
      : _last_duration(duration_type())
      , _accumulated_duration(duration_type())
      , _accumulation_count(0u)
    {}
    virtual ~accum_timer_base() {}

    virtual void                    stop()  = 0;
    virtual void                    collect() = 0;
    virtual void                    force_collect() = 0;
    virtual void                    reset() {
        _last_duration          = duration_type();
        _accumulated_duration   = duration_type();
        _accumulation_count     = 0u;
    }

    const duration_type&            accumulated_duration() const { return _accumulated_duration; }
    unsigned                        accumulation_count() const   { return _accumulation_count; }

    duration_type                   average_duration() const {
        if (_accumulation_count > 0) {
            return _accumulated_duration / _accumulation_count;
        }
        else {
            return duration_type();
        }
    }

    duration_type                   last_time() const { return _last_duration; }

protected:
    duration_type                   _last_duration;
    duration_type                   _accumulated_duration;
    unsigned                        _accumulation_count;

}; // class accum_timer_base

template<class timer_t>
class accum_timer : public accum_timer_base
{
public:
    typedef timer_t         timer_type;

public:
    accum_timer();
    virtual ~accum_timer();

    void                    start();
    void                    stop();
    void                    collect();
    void                    force_collect();

protected:
    timer_type              _timer;

}; // class accum_timer

} // namespace time
} // namespace scm

#include "accum_timer.inl"

#endif // SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED
