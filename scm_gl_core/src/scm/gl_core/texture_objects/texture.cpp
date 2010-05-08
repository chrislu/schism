
#include "texture.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

namespace util {
} // namespace util

texture::texture(render_device& in_device)
  : render_target(in_device)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();

    glapi.glGenTextures(1, &_gl_object_id);
    if (0 == _gl_object_id) {
        state().set(object_state::OS_BAD);
    }

    gl_assert(glapi, leaving texture::texture());
}

texture::~texture()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_object_id);
    glapi.glDeleteTextures(1, &_gl_object_id);
    
    gl_assert(glapi, leaving texture::~texture());
}

void
texture::bind(const render_context& in_context, int in_unit) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != texture_binding());

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    glapi.glBindMultiTextureEXT(GL_TEXTURE0 + in_unit, object_target(), object_id());
    //glapi.glEnableIndexedEXT(object_target(), _occupied_texture_unit);
#else  // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    glapi.glActiveTexture(GL_TEXTURE0 + in_unit);
    glapi.glBindTexture(object_target(), object_id());
    //glapi.glEnable(object_target());
    glapi.glActiveTexture(GL_TEXTURE0);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    gl_assert(glapi, leaving texture::bind());
}

void
texture::unbind(const render_context& in_context, int in_unit) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != texture_binding());

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    glapi.glBindMultiTextureEXT(GL_TEXTURE0 + in_unit, object_target(), 0);
    //glapi.glDisableIndexedEXT(object_target(), _occupied_texture_unit);
#else  // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    glapi.glActiveTexture(GL_TEXTURE0 + in_unit);
    glapi.glBindTexture(object_target(), 0);
    //glapi.glDisable(object_target());
    glapi.glActiveTexture(GL_TEXTURE0);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    gl_assert(glapi, leaving texture::unbind());
}

unsigned
texture::texture_binding() const
{
    return (_gl_texture_binding);
}

} // namespace gl
} // namespace scm
