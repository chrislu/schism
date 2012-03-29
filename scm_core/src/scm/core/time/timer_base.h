
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_TIME_TIMER_BASE_H_INCLUDED
#define SCM_CORE_TIME_TIMER_BASE_H_INCLUDED

#include <iosfwd>
#include <string>

#include <boost/timer/timer.hpp>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) timer_base
{
public:
    typedef boost::timer::nanosecond_type   nanosec_type;

    enum time_unit {
        sec,
        msec,
        usec,
        nsec
    };
    enum throughput_unit {
        Bps,
        KiBps,
        MiBps,
        GiBps
    };

public:
    timer_base();
    virtual ~timer_base();

    virtual void                start()                  = 0;
    virtual void                stop()                   = 0;
    virtual bool                result_available() const = 0;
    virtual void                collect() const          = 0;
    virtual void                force_collect() const    = 0;

    virtual nanosec_type        elapsed() const          = 0;

    virtual void                report(std::ostream&   os,
                                       time_unit       tunit  = msec,
                                       size_t          dsize  = 0,
                                       throughput_unit tpunit = MiBps) const = 0;

protected:
    double                      to_time_unit(time_unit tu, nanosec_type t) const;
    std::string                 time_unit_string(time_unit tu) const;

}; // class timer_base

} // namespace time
} // namespace scm

#endif // SCM_CORE_TIME_TIMER_BASE_H_INCLUDED
