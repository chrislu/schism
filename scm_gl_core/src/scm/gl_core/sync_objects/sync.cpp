
#include "sync.h"

#include <cassert>

#include <scm/gl_core/config.h>
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
  , _gl_sync_object(nullptr)
{
}

sync::~sync()
{
    delete_sync();
}

sync::GLsync
sync::gl_object() const
{
    return _gl_sync_object;
}

void
sync::delete_sync()
{
    if (_gl_sync_object) {
        const opengl::gl_core& glapi = parent_device().opengl_api();
        glapi.glDeleteSync(_gl_sync_object);
        _gl_sync_object = nullptr;
    }
    
    gl_assert(glapi, leaving sync::~delete_sync());
}

} // namespace gl
} // namespace scm
