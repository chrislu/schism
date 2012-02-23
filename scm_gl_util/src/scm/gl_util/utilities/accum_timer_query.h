
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_accum_timer_query_H_INCLUDED
#define SCM_GL_UTIL_accum_timer_query_H_INCLUDED

#include <scm/core/time/accum_timer.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) accum_timer_query : public time::accum_timer_base
{
public:
    accum_timer_query(const render_device_ptr& device);
    virtual ~accum_timer_query();

    void                    start(const render_context_ptr& context);
    void                    stop();
    void                    collect();
    void                    force_collect();
    void                    reset();

protected:
    timer_query_ptr         _timer_query_begin;
    timer_query_ptr         _timer_query_end;
    bool                    _timer_query_finished;

    render_context_ptr      _timer_context;

}; // class accum_timer_query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_accum_timer_query_H_INCLUDED
