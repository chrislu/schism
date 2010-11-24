
#include "texture.h"

#include <cassert>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace scm {
namespace gl {

namespace util {
} // namespace util

texture::texture(render_device& in_device)
  : context_bindable_object()
  , render_device_resource(in_device)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();

    glapi.glGenTextures(1, &(context_bindable_object::_gl_object_id));
    if (0 == _gl_object_id) {
        state().set(object_state::OS_BAD);
    }

    gl_assert(glapi, leaving texture::texture());
}

texture::~texture()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_object_id);
    glapi.glDeleteTextures(1, &(context_bindable_object::_gl_object_id));
    
    gl_assert(glapi, leaving texture::~texture());
}

void
texture::bind(const render_context& in_context, int in_unit) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != object_binding());

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glBindMultiTextureEXT(GL_TEXTURE0 + in_unit, object_target(), object_id());
        //glapi.glEnableIndexedEXT(object_target(), _occupied_texture_unit);
    }
    else {
        glapi.glActiveTexture(GL_TEXTURE0 + in_unit);
        glapi.glBindTexture(object_target(), object_id());

        if (SCM_GL_CORE_USE_WORKAROUND_AMD) {
            // AMD bug!!! overwritten by sampler object anyway
            glapi.glTexParameteri(object_target(), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glapi.glTexParameteri(object_target(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        //glapi.glTexParameteri(object_target(), GL_TEXTURE_SWIZZLE_R, GL_RED);
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_SWIZZLE_G, GL_GREEN);
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_SWIZZLE_B, GL_BLUE);
        //glapi.glTexParameteri(object_target(), GL_TEXTURE_SWIZZLE_A, GL_ALPHA);

        //glapi.glEnable(object_target());
        glapi.glActiveTexture(GL_TEXTURE0);
    }

    gl_assert(glapi, leaving texture::bind());
}

void
texture::unbind(const render_context& in_context, int in_unit) const
{
    const opengl::gl3_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != object_binding());

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glBindMultiTextureEXT(GL_TEXTURE0 + in_unit, object_target(), 0);
        //glapi.glDisableIndexedEXT(object_target(), _occupied_texture_unit);
    }
    else {
        glapi.glActiveTexture(GL_TEXTURE0 + in_unit);
        glapi.glBindTexture(object_target(), 0);
        //glapi.glDisable(object_target());
        glapi.glActiveTexture(GL_TEXTURE0);
    }

    gl_assert(glapi, leaving texture::unbind());
}

} // namespace gl
} // namespace scm
