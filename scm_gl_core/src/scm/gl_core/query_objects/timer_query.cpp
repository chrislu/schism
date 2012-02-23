
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "timer_query.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

timer_query::timer_query(render_device& in_device)
  : query(in_device),
    _result(0)
{
    _gl_query_type = GL_TIME_ELAPSED;
    // start and stop the query to actually generate the query object
    begin(*in_device.main_context());
    end(*in_device.main_context());
}

timer_query::~timer_query()
{
}

void
timer_query::query_counter(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != query_type());

    glapi.glQueryCounter(query_id(), GL_TIMESTAMP);

    gl_assert(glapi, leaving timer_query::collect());
}

void
timer_query::collect(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != query_type());

    glapi.glGetQueryObjectui64v(query_id(), GL_QUERY_RESULT, &_result);

    gl_assert(glapi, leaving timer_query::collect());
}

scm::uint64
timer_query::result() const
{
    return _result;
}

} // namespace gl
} // namespace scm
