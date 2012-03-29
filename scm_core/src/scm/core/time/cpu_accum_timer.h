
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_TIME_CPU_ACCUM_TIMER_H_INCLUDED
#define SCM_CORE_TIME_CPU_ACCUM_TIMER_H_INCLUDED

#include <scm/core/time/accum_timer_base.h>
#include <scm/core/time/cpu_timer.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) cpu_accum_timer : public accum_timer_base
{
public:
    typedef cpu_timer::cpu_times    cpu_times;

public:
    cpu_accum_timer();
    virtual ~cpu_accum_timer();

    void                            start();
    void                            stop();
    void                            collect();
    void                            force_collect();

    void                            update(int interval = 100);
    void                            reset();

    cpu_times                       detailed_last_time() const;
    cpu_times                       detailed_accumulated_time() const;
    cpu_times                       detailed_average_time() const;

    void                            report(std::ostream& os,               time_io unit = time_io(time_io::msec))                 const;
    void                            report(std::ostream& os, size_t dsize, time_io unit = time_io(time_io::msec, time_io::MiBps)) const;
    void                            detailed_report(std::ostream& os,               time_io unit  = time_io(time_io::msec))                 const;
    void                            detailed_report(std::ostream& os, size_t dsize, time_io unit  = time_io(time_io::msec, time_io::MiBps)) const;

protected:
    cpu_times                       _detailed_last_time;
    cpu_times                       _detailed_accumulated_time;
    cpu_times                       _detailed_average_time;
    cpu_timer                       _cpu_timer;

}; // class cpu_accum_timer

} // namespace time
} // namespace scm

#endif // SCM_CORE_TIME_CPU_ACCUM_TIMER_H_INCLUDED
