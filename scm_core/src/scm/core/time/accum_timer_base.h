
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_TIME_ACCUM_TIMER_BASE_H_INCLUDED
#define SCM_CORE_TIME_ACCUM_TIMER_BASE_H_INCLUDED

#include <scm/core/time/timer_base.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) accum_timer_base
{
public:
    typedef timer_base::nanosec_type    nanosec_type;

public:
    accum_timer_base();
    virtual ~accum_timer_base();

    virtual void                    stop()          = 0;
    virtual void                    collect()       = 0;
    virtual void                    force_collect() = 0;

    virtual void                    update(int interval = 100);
    virtual void                    reset();

    nanosec_type                    last_time() const;
    nanosec_type                    accumulated_time() const;
    unsigned                        accumulation_count() const;
    nanosec_type                    average_time() const;

    double                          last_time(time_io::time_unit tu) const;
    double                          accumulated_time(time_io::time_unit tu) const;
    double                          average_time(time_io::time_unit tu) const;

    virtual void                    report(std::ostream& os,               time_io unit = time_io(time_io::msec))                 const = 0;
    virtual void                    report(std::ostream& os, size_t dsize, time_io unit = time_io(time_io::msec, time_io::MiBps)) const = 0;
    virtual void                    detailed_report(std::ostream& os,               time_io unit  = time_io(time_io::msec))                 const = 0;
    virtual void                    detailed_report(std::ostream& os, size_t dsize, time_io unit  = time_io(time_io::msec, time_io::MiBps)) const = 0;

protected:
    nanosec_type                    _last_time;
    nanosec_type                    _accumulated_time;
    nanosec_type                    _average_time;
    unsigned                        _accumulation_count;
    int                             _update_interval;

}; // class accum_timer_base

} // namespace time
} // namespace scm

#endif // SCM_CORE_TIME_ACCUM_TIMER_BASE_H_INCLUDED
