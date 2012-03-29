
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

typedef boost::timer::nanosecond_type   nanosec_type;

struct __scm_export(core) time_io {
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

    time_unit           _t_unit;
    int                 _t_dec_places;
    throughput_unit     _tp_unit;
    int                 _tp_dec_places;

    time_io(time_unit tu, int tdec = 2) : _t_unit(tu), _t_dec_places(tdec), _tp_unit(Bps), _tp_dec_places(1) {}
    time_io(time_unit tu, throughput_unit tpu, int tdec = 2, int tpdec = 2) : _t_unit(tu), _t_dec_places(tdec), _tp_unit(tpu), _tp_dec_places(tpdec) {}

    static double               to_time_unit(time_unit tu, nanosec_type t);
    static double               to_throughput_unit(throughput_unit tu, nanosec_type t, size_t d);
    static std::string          time_unit_string(time_unit tu);
    static std::string          throughput_unit_string(throughput_unit tu);
};


class __scm_export(core) timer_base
{
public:
    typedef scm::time::nanosec_type   nanosec_type;

public:
    timer_base();
    virtual ~timer_base();

    virtual void                start()                  = 0;
    virtual void                stop()                   = 0;
    virtual bool                result_available() const = 0;
    virtual void                collect() const          = 0;
    virtual void                force_collect() const    = 0;

    virtual nanosec_type        elapsed() const          = 0;

    virtual void                report(std::ostream& os,               time_io unit = time_io(time_io::msec))                 const = 0;
    virtual void                report(std::ostream& os, size_t dsize, time_io unit = time_io(time_io::msec, time_io::MiBps)) const = 0;
    virtual void                detailed_report(std::ostream& os,               time_io unit  = time_io(time_io::msec))                 const = 0;
    virtual void                detailed_report(std::ostream& os, size_t dsize, time_io unit  = time_io(time_io::msec, time_io::MiBps)) const = 0;

    double                      elapsed(time_io::time_unit tu) const;

protected:

}; // class timer_base

} // namespace time
} // namespace scm

#endif // SCM_CORE_TIME_TIMER_BASE_H_INCLUDED
