
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.
#include "sync.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/constants.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

sync::sync(render_device& in_device)
  : render_device_child(in_device)
  , _gl_sync_object(0)//nullptr)
{
}

sync::~sync()
{
    delete_sync();
}

sync::GLsync
sync::object() const
{
    return _gl_sync_object;
}

sync_wait_result
sync::client_wait(const render_context& in_context,
                        scm::uint64     in_timeout,
                        bool            in_flush) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object());

    GLenum r = glapi.glClientWaitSync(object(), in_flush ? GL_SYNC_FLUSH_COMMANDS_BIT : 0, in_timeout);

    gl_assert(glapi, leaving sync::client_wait());

    switch (r) {
        case GL_ALREADY_SIGNALED:       return SYNC_WAIT_ALREADY_SIGNALED;
        case GL_CONDITION_SATISFIED:    return SYNC_WAIT_CONDITION_SATISFIED;
        case GL_TIMEOUT_EXPIRED:        return SYNC_WAIT_TIMEOUT_EXPIRED;
        case GL_WAIT_FAILED:
        default:                        return SYNC_WAIT_FAILED;
    }
}

void
sync::server_wait(const render_context& in_context,
                        scm::uint64     in_timeout) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object());

    glapi.glWaitSync(object(), 0, sync_timeout_ignored);

    gl_assert(glapi, leaving sync::server_wait());
}

sync_status
sync::status(const render_context& in_context) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object());

    int cur_status = 0;

    glapi.glGetSynciv(object(), GL_SYNC_STATUS, 1, 0, &cur_status);

    switch (cur_status) {
        case GL_SIGNALED:   return SYNC_SIGNALED;
        case GL_UNSIGNALED:
        default:            return SYNC_UNSIGNALED;
    }

    gl_assert(glapi, leaving sync::status());
}

void
sync::delete_sync()
{
    if (_gl_sync_object) {
        const opengl::gl_core& glapi = parent_device().opengl_api();
        glapi.glDeleteSync(_gl_sync_object);
        _gl_sync_object = 0;//nullptr;

        gl_assert(glapi, leaving sync::~delete_sync());
    }
}

} // namespace gl
} // namespace scm
