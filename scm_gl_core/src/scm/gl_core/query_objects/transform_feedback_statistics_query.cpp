
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "transform_feedback_statistics_query.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

transform_feedback_statistics::transform_feedback_statistics()
  : _primitives_generated(0)
  , _primitives_written(0)
{
}

transform_feedback_statistics_query::transform_feedback_statistics_query(render_device& in_device, int stream)
  : query(in_device)
  , _result()
{
    _gl_query_type          = GL_PRIMITIVES_GENERATED;
    _query_type_xfb_written = GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN;

    const opengl::gl_core& glapi = in_device.opengl_api();

    if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
        // GL3.x
        if (stream > 0) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            SCM_GL_DGB("transform_feedback_statistics_query::transform_feedback_statistics_query(): error, only OpenGL 4+ allows for indexed transform feedback queries)");
            return;
        }
    }
    else {
        // GL4.x+
        if (stream > in_device.capabilities()._max_vertex_streams) {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            SCM_GL_DGB("transform_feedback_statistics_query::transform_feedback_statistics_query(): error, unsupported stream number "
                       << "(max. supported streams: " << in_device.capabilities()._max_vertex_streams << ")");
            return;
        }
    }
    _index = stream;

    glapi.glGenQueries(1, &_query_id_xfb_written);
    if (0 == _query_id_xfb_written) {
        state().set(object_state::OS_BAD);
    }

    gl_assert(glapi, leaving transform_feedback_statistics_query::transform_feedback_statistics_query());

}

transform_feedback_statistics_query::~transform_feedback_statistics_query()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();

    assert(0 != _query_id_xfb_written);
    glapi.glDeleteQueries(1, &_query_id_xfb_written);
    
    gl_assert(glapi, leaving transform_feedback_statistics_query::~transform_feedback_statistics_query());
}

void
transform_feedback_statistics_query::begin(const render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != _query_id_xfb_written);
    assert(0 != query_type());
    assert(0 != _query_type_xfb_written);

    if (index() > 0) {
        if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
            glapi.glBeginQueryIndexed(query_type(),            index(), query_id());
            glapi.glBeginQueryIndexed(_query_type_xfb_written, index(), _query_id_xfb_written);
        }
        else {
            SCM_GL_DGB("transform_feedback_statistics_query::begin(): error, only OpenGL 4+ allows for indexed transform feedback queries)");
        }
    }
    else {
        glapi.glBeginQuery(query_type(), query_id());
        glapi.glBeginQuery(_query_type_xfb_written, _query_id_xfb_written);
    }

    gl_assert(glapi, leaving query::begin());
}

void
transform_feedback_statistics_query::end(const render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != _query_id_xfb_written);
    assert(0 != query_type());
    assert(0 != _query_type_xfb_written);

    if (index() > 0) {
        if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400) {
            glapi.glEndQueryIndexed(query_type(),            index());
            glapi.glEndQueryIndexed(_query_type_xfb_written, index());
        }
        else {
            SCM_GL_DGB("transform_feedback_statistics_query::begin(): error, only OpenGL 4+ allows for indexed transform feedback queries)");
        }
    }
    else {
        glapi.glEndQuery(query_type());
        glapi.glEndQuery(_query_type_xfb_written);
    }

    gl_assert(glapi, leaving query::end());
}

void
transform_feedback_statistics_query::collect(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != _query_id_xfb_written);
    assert(0 != query_type());
    assert(0 != _query_type_xfb_written);

    glapi.glGetQueryObjectiv(query_id(),            GL_QUERY_RESULT, &_result._primitives_generated);
    glapi.glGetQueryObjectiv(_query_id_xfb_written, GL_QUERY_RESULT, &_result._primitives_written);

    gl_assert(glapi, leaving transform_feedback_statistics_query::collect());
}

const transform_feedback_statistics&
transform_feedback_statistics_query::result() const
{
    return _result;
}

} // namespace gl
} // namespace scm
