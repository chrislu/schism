
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_accum_timer_query_H_INCLUDED
#define SCM_GL_UTIL_accum_timer_query_H_INCLUDED

#include <scm/core/time/accum_timer_base.h>
#include <scm/core/time/cpu_timer.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) accum_timer_query : public time::accum_timer_base
{
public:
    typedef time::cpu_timer::nanosec_type   nanosec_type;
    struct gl_times : public time::cpu_timer::cpu_times
    {
        nanosec_type    gl;
    };

public:
    accum_timer_query(const render_device_ptr& device);
    virtual ~accum_timer_query();

    void                    start(const render_context_ptr& context);
    void                    stop();
    void                    collect();
    void                    force_collect();

    void                    update(int interval = 100);
    void                    reset();

    gl_times                detailed_last_time() const;
    gl_times                detailed_accumulated_time() const;
    gl_times                detailed_average_time() const;

    void                    report(std::ostream& os,               time::time_io unit = time::time_io(time::time_io::msec))                       const;
    void                    report(std::ostream& os, size_t dsize, time::time_io unit = time::time_io(time::time_io::msec, time::time_io::MiBps)) const;
    void                    detailed_report(std::ostream& os,               time::time_io unit  = time::time_io(time::time_io::msec))                       const;
    void                    detailed_report(std::ostream& os, size_t dsize, time::time_io unit  = time::time_io(time::time_io::msec, time::time_io::MiBps)) const;

protected:
    timer_query_ptr         _timer_query_begin;
    timer_query_ptr         _timer_query_end;
    bool                    _timer_query_finished;

    render_context_ptr      _timer_context;

    gl_times                _detailed_last_time;
    gl_times                _detailed_accumulated_time;
    gl_times                _detailed_average_time;
    time::cpu_timer         _cpu_timer;

}; // class accum_timer_query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_accum_timer_query_H_INCLUDED
