
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_TIME_CPU_TIMER_H_INCLUDED
#define SCM_CORE_TIME_CPU_TIMER_H_INCLUDED

#include <scm/core/time/timer_base.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) cpu_timer : public timer_base
{
public:
    typedef boost::timer::cpu_times cpu_times;

public:
    cpu_timer();
    virtual ~cpu_timer();

    void                start();
    void                stop();
    bool                result_available() const;
    void                collect() const;
    void                force_collect() const;

    nanosec_type        elapsed() const;
    cpu_times           detailed_elapsed() const;

    void                report(std::ostream& os,               time_io unit = time_io(time_io::msec))                 const;
    void                report(std::ostream& os, size_t dsize, time_io unit = time_io(time_io::msec, time_io::MiBps)) const;
    void                detailed_report(std::ostream& os,               time_io unit  = time_io(time_io::msec))                 const;
    void                detailed_report(std::ostream& os, size_t dsize, time_io unit  = time_io(time_io::msec, time_io::MiBps)) const;

protected:
    boost::timer::cpu_timer _timer;

}; // class cpu_timer

} // namespace time
} // namespace scm

#endif // SCM_CORE_TIME_CPU_TIMER_H_INCLUDED
