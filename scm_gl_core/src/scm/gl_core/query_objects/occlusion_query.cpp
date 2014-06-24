
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "occlusion_query.h"

#include <cassert>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

occlusion_query::occlusion_query(render_device& in_device, const occlusion_query_mode in_oq_mode)
  : query(in_device),
    _result(0)
{
    switch(in_oq_mode) {
        case OQMODE_SAMPLES_PASSED: {
                    _gl_query_type = GL_SAMPLES_PASSED;
                    break;
        }
        case OQMODE_ANY_SAMPLES_PASSED: {
            if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_330) {
                // < GL3.3
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                SCM_GL_DGB("occlusion_query::occlusion_query(): error, only OpenGL 3.3+ allows for occlusion query mode ANY_SAMPLES_PASSED)");
            return;
            }
            _gl_query_type = GL_ANY_SAMPLES_PASSED;
            break;
        }
        case OQMODE_ANY_SAMPLES_PASSED_CONSERVATIVE: {
            if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_430) {
                // < GL4.3
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                SCM_GL_DGB("occlusion_query::occlusion_query(): error, only OpenGL 4.3+ allows for occlusion query mode ANY_SAMPLES_PASSED_CONSERVATIVE)");
            return;
            }
            _gl_query_type = GL_ANY_SAMPLES_PASSED_CONSERVATIVE;
            break;
        }
        default: { 
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            SCM_GL_DGB("occlusion_query()::occlusion_query(): error, unknown occlusion query mode)");
            return;
        }
    }


    // start and stop the query to actually generate the query object
    begin(*in_device.main_context());
    end(*in_device.main_context());
}

occlusion_query::~occlusion_query()
{
}

void
occlusion_query::collect(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != query_id());
    assert(0 != query_type());

    glapi.glGetQueryObjectui64v(query_id(), GL_QUERY_RESULT, &_result);

    gl_assert(glapi, leaving occlusion_query::collect());
}

scm::uint64
occlusion_query::result() const
{
    return _result;
}

} // namespace gl
} // namespace scm
