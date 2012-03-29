
#include "fence_sync.h"

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

fence_sync::fence_sync(render_device& in_device)
  : sync(in_device)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    _gl_sync_object = glapi.glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!_gl_sync_object) {
        state().set(object_state::OS_BAD);
    }

    gl_assert(glapi, leaving texture::texture());
}

fence_sync::~fence_sync()
{
}

} // namespace gl
} // namespace scm
