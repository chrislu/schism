
#include "texture_image.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

namespace util {
} // namespace util

texture_image::texture_image(render_device& in_device)
  : texture(in_device)
  , render_target()
{
}

texture_image::~texture_image()
{
}

void
texture_image::generate_mipmaps(const render_context& in_context)
{
    const opengl::gl3_core& glapi = in_context.opengl_api();

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glGenerateTextureMipmapEXT(object_id(), object_target());
    }
    else {
        util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
        glapi.glBindTexture(object_target(), object_id());
        glapi.glGenerateMipmap(object_target());
    }

    gl_assert(glapi, leaving texture::generate_mipmaps());
}

unsigned
texture_image::object_id() const
{
    return context_bindable_object::object_id();
}

unsigned
texture_image::object_target() const
{
    return context_bindable_object::object_target();
}


} // namespace gl
} // namespace scm
