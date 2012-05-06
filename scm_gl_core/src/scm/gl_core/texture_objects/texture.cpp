
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture.h"

#include <cassert>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

#include <scm/gl_core/state_objects/sampler_state.h>

namespace scm {
namespace gl {

namespace util {
} // namespace util

texture::texture(render_device& in_device)
  : context_bindable_object()
  , render_device_resource(in_device)
  , _native_handle(0ull)
  , _native_handle_resident(false)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    glapi.glGenTextures(1, &(context_bindable_object::_gl_object_id));
    if (0 == _gl_object_id) {
        state().set(object_state::OS_BAD);
    }

    gl_assert(glapi, leaving texture::texture());
}

texture::~texture()
{
    const opengl::gl_core& glapi = parent_device().opengl_api();
    gl_assert(glapi, entering texture::~texture());

    if (   glapi.extension_NV_bindless_texture
        && _native_handle
        && _native_handle_resident) {
        glapi.glMakeTextureHandleNonResidentNV(_native_handle);
        gl_assert(glapi, texture::~texture() after glMakeTextureHandleNonResidentNV);
    }

    assert(0 != _gl_object_id);
    glapi.glDeleteTextures(1, &(context_bindable_object::_gl_object_id));
    
    gl_assert(glapi, leaving texture::~texture());
}

uint64
texture::native_handle() const
{
    return _native_handle;
}

bool
texture::native_handle_resident() const
{
    return _native_handle_resident;
}

void
texture::bind(const render_context& in_context, int in_unit) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
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
    const opengl::gl_core& glapi = in_context.opengl_api();
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

void
texture::bind_image(const render_context& in_context,
                          unsigned        in_unit,
                          data_format     in_format,
                          access_mode     in_access,
                          int             in_level,
                          int             in_layer) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != object_binding());
    
    // TODO runtime checks for level and layer, maybe format compatibility
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {
        glapi.glBindImageTexture(in_unit, object_id(),
                                 in_level, (in_layer >= 0), in_layer >= 0 ? in_layer : 0,
                                 util::gl_image_access_mode(in_access),
                                 util::gl_internal_format(in_format));
    }
    else { // fall back to EXT_shader_image_load_store
        assert(glapi.extension_EXT_shader_image_load_store == true);

        glapi.glBindImageTextureEXT(in_unit, object_id(),
                                    in_level, (in_layer >= 0), in_layer >= 0 ? in_layer : 0,
                                    util::gl_image_access_mode(in_access),
                                    util::gl_internal_format(in_format));
    }

    gl_assert(glapi, leaving texture::unbind());
}

void
texture::unbind_image(const render_context& in_context, int in_unit) const
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    assert(0 != object_id());
    assert(0 != object_target());
    assert(0 != object_binding());

    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {
        glapi.glBindImageTexture(in_unit, 0,
                                 0, false, 0,
                                 GL_READ_WRITE,
                                 GL_RGBA8);
    }
    else { // fall back to EXT_shader_image_load_store
        assert(glapi.extension_EXT_shader_image_load_store == true);

        glapi.glBindImageTextureEXT(in_unit, 0,
                                    0, false, 0,
                                    GL_READ_WRITE,
                                    GL_RGBA8);
    }
    gl_assert(glapi, leaving texture::unbind());
}

bool
texture::make_resident(const render_context&    in_context,
                       const sampler_state_ptr& in_sstate)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    util::gl_error         glerror(glapi);

    if (!glapi.extension_NV_bindless_texture) {
        return false;
    }

    if (   _native_handle
        && _native_handle_resident) {
        if (!make_non_resident(in_context)) {
            glerr() << log::error
                    << "texture::make_resident() error making current resident handle non-resident (NV_bindless_texture).";
            return false;
        }
    }

    _native_handle = glapi.glGetTextureSamplerHandleNV(object_id(), in_sstate->sampler_id());

    if (glerror || 0ull == _native_handle) {
        glerr() << log::error
                << "texture::make_resident() error getting texture/sampler handle (NV_bindless_texture): "
                << glerror.error_string();
        return false;
    }

    glapi.glMakeTextureHandleResidentNV(_native_handle);

    if (glerror) {
        glerr() << log::error
                << "texture::make_resident() error making texture handle resident (NV_bindless_texture): "
                << glerror.error_string();
        return false;
    }

    _native_handle_resident = true;

    return true;
}

bool
texture::make_non_resident(const render_context& in_context)
{
    const opengl::gl_core& glapi = in_context.opengl_api();
    util::gl_error         glerror(glapi);

    if (!glapi.extension_NV_bindless_texture) {
        return false;
    }

    if (   _native_handle
        && _native_handle_resident) {
        glapi.glMakeTextureHandleNonResidentNV(_native_handle);
        
        if (glerror) {
            glerr() << log::error
                    << "texture::make_non_resident() error making texture handle non-resident (NV_bindless_texture): "
                    << glerror.error_string();
            return false;
        }
        _native_handle_resident = false;
    }

    return true;
}

} // namespace gl
} // namespace scm
