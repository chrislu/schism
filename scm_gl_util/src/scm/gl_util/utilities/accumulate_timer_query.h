
#ifndef SCM_GL_UTIL_ACCUMULATE_TIMER_QUERY_H_INCLUDED
#define SCM_GL_UTIL_ACCUMULATE_TIMER_QUERY_H_INCLUDED

#include <scm/core/time/time_types.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) accumulate_timer_query
{
public:
    typedef time::time_duration      duration_type;

public:
    accumulate_timer_query(const render_device_ptr& device);

    void                    start(const render_context_ptr& context);
    void                    stop(const render_context_ptr& context);
    void                    collect(const render_context_ptr& context);
    void                    reset();

    const duration_type&    accumulated_duration() const;
    unsigned                accumulation_count() const;

    duration_type           average_duration() const;

protected:
    duration_type           _accumulated_duration;
    unsigned                _accumulation_count;

    timer_query_ptr         _timer_query;

}; // class accumulate_timer_query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_ACCUMULATE_TIMER_QUERY_H_INCLUDED
