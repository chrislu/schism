
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED
#define SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED

#include <scm/core/memory.h>
#include <scm/core/time/time_types.h>

namespace scm {
namespace time {

//class accum_timer_base
//{
//public:
//    typedef boost::timer::cpu_times         cpu_times;
//    typedef boost::timer::nanosecond_type   nanosec_type;
//
//public:
//    accum_timer_base() 
//    {
//        reset();
//    }
//    virtual ~accum_timer_base() {}
//
//    virtual void                    stop()  = 0;
//    virtual void                    collect() = 0;
//    virtual void                    force_collect() = 0;
//    virtual void                    reset() {
//        _last_times.wall          = 0;
//        _last_times.user          = 0;
//        _last_times.system        = 0;
//        _accumulated_times.wall   = 0;
//        _accumulated_times.user   = 0;
//        _accumulated_times.system = 0;
//        _accumulation_count       = 0u;
//    }
//
//    cpu_times                       last_times()         const   { return _last_times; }
//    cpu_times                       accumulated_times()  const   { return _accumulated_times; }
//    unsigned                        accumulation_count() const   { return _accumulation_count; }
//
//    cpu_times                       average_times() const {
//        cpu_times avg;
//        avg.wall = avg.user = avg.system = 0;
//        if (_accumulation_count > 0) {
//            avg.wall   = _accumulated_times.wall   / _accumulation_count;
//            avg.user   = _accumulated_times.user   / _accumulation_count;
//            avg.system = _accumulated_times.system / _accumulation_count;
//        }
//        return avg;
//    }
//
//protected:
//    cpu_times                       _last_times;
//    cpu_times                       _accumulated_times;
//    unsigned                        _accumulation_count;
//
//}; // class accum_timer_base

class accum_timer_base_deprecated
{
public:
    typedef time::time_duration     duration_type;

public:
    accum_timer_base_deprecated() 
      : _last_duration(duration_type())
      , _accumulated_duration(duration_type())
      , _accumulation_count(0u)
    {}
    virtual ~accum_timer_base_deprecated() {}

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

}; // class accum_timer_base_deprecated

template<class timer_t>
class accum_timer : public accum_timer_base_deprecated
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
