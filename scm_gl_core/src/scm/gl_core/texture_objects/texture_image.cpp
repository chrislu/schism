
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_image.h"

#include <cassert>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
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
    const opengl::gl_core& glapi = in_context.opengl_api();

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

bool
texture_image::retrieve_image_data(const render_context& in_context,
                                   const unsigned        in_level,
                                         void*           in_data)
{
    const opengl::gl_core& glapi = in_context.opengl_api();

    util::gl_error         glerror(glapi);
    
    unsigned gl_internal_format = util::gl_internal_format(format());
    unsigned gl_base_format     = util::gl_base_format(format());
    unsigned gl_base_type       = util::gl_base_type(format());

    if (in_level >= mip_map_layers()) {
        return false;
    }
    if (samples() > 1) {
        return false;
    }

    if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
        glapi.glGetTextureImageEXT(object_id(), object_target(), in_level, gl_base_format, gl_base_type, in_data);
    }
    else {
        util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
        glapi.glBindTexture(object_target(), object_id());

        glapi.glGetTexImage(object_target(), in_level, gl_base_format, gl_base_type, in_data);
    }

    gl_assert(glapi, leaving texture::generate_mipmaps());

    if (glerror) {
        //state().set(glerror.to_object_state());
        return false;
    }
    else {
        return true;
    }
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
