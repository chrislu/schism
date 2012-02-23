
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TRANSFORM_FEEDBACK_STATISTICS_QUERY_H_INCLUDED
#define SCM_GL_CORE_TRANSFORM_FEEDBACK_STATISTICS_QUERY_H_INCLUDED

#include <scm/core/numeric_types.h>

#include <scm/gl_core/query_objects/query.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) transform_feedback_statistics
{
    int _primitives_generated;
    int _primitives_written;

    transform_feedback_statistics();
}; // struct transform_feedback_statistics

class __scm_export(gl_core) transform_feedback_statistics_query : public query
{
public:
    virtual ~transform_feedback_statistics_query();

    const transform_feedback_statistics&    result() const;

protected:
    transform_feedback_statistics_query(render_device& in_device, int stream = 0);

    void            begin(const render_context& in_context) const;
    void            end(const render_context& in_context) const;
    void            collect(const render_context& in_context);

protected:
    transform_feedback_statistics           _result;

    unsigned        _query_id_xfb_written;
    unsigned        _query_type_xfb_written;

private:
    friend class render_device;
    friend class render_context;
}; // class transform_feedback_statistics_query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TRANSFORM_FEEDBACK_STATISTICS_QUERY_H_INCLUDED
