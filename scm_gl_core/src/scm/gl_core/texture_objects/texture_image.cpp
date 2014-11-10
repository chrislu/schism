
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_image.h"

#include <cassert>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
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

    // NO blocking error test here!
    return true;
#if 0
    if (glerror) {
        //state().set(glerror.to_object_state());
        return false;
    }
    else {
        return true;
    }
#endif
}

bool
texture_image::clear_image_data(const render_context& in_context,
                                const unsigned        in_level,
                                const data_format     in_data_format,
                                const void*const      in_data)
{
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
    const opengl::gl_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering texture_image::clear_image_data());

    util::gl_error glerror(glapi);

    unsigned gl_base_format      = util::gl_base_format(in_data_format);
    unsigned gl_base_type        = util::gl_base_type(in_data_format);
    unsigned gl_this_base_format = util::gl_base_format(format());
    unsigned gl_this_base_type   = util::gl_base_type(format());

    if (is_compressed_format(format())) {
        glerr() << log::error
                << "texture_image::clear_image_data(): currently not supporting compressed data formats" << log::end;
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return false;
    }

    if (   gl_this_base_format == GL_DEPTH_COMPONENT
        && gl_base_format      != GL_DEPTH_COMPONENT)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (   gl_this_base_format == GL_DEPTH_STENCIL
        && gl_base_format      != GL_DEPTH_STENCIL)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (   gl_this_base_format == GL_STENCIL_INDEX
        && gl_base_format      != GL_STENCIL_INDEX)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    if (is_integer_type(format()) && !is_integer_type(in_data_format)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (!is_integer_type(format()) && is_integer_type(in_data_format)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    glapi.glClearTexImage(object_id(), in_level, gl_base_format, gl_base_type, in_data);

    gl_assert(glapi, leaving texture_image::clear_image_data());
    return true;

#else // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
    return false;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
}

bool
texture_image::clear_image_sub_data(const render_context& in_context,
                                    const texture_region& in_region,
                                    const unsigned        in_level,
                                    const data_format     in_data_format,
                                    const void*const      in_data)
{
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
    const opengl::gl_core& glapi = in_context.opengl_api();

    gl_assert(glapi, entering texture_image::clear_image_sub_data());

    util::gl_error glerror(glapi);

    unsigned gl_base_format      = util::gl_base_format(in_data_format);
    unsigned gl_base_type        = util::gl_base_type(in_data_format);
    unsigned gl_this_base_format = util::gl_base_format(format());
    unsigned gl_this_base_type   = util::gl_base_type(format());

    if (is_compressed_format(format())) {
        glerr() << log::error
                << "texture_image::clear_image_sub_data(): currently not supporting compressed data formats" << log::end;
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return false;
    }

    if (   gl_this_base_format == GL_DEPTH_COMPONENT
        && gl_base_format      != GL_DEPTH_COMPONENT)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (   gl_this_base_format == GL_DEPTH_STENCIL
        && gl_base_format      != GL_DEPTH_STENCIL)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (   gl_this_base_format == GL_STENCIL_INDEX
        && gl_base_format      != GL_STENCIL_INDEX)
    {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    if (is_integer_type(format()) && !is_integer_type(in_data_format)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (!is_integer_type(format()) && is_integer_type(in_data_format)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    glapi.glClearTexSubImage(object_id(),
                             in_level,
                             in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                             in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                             gl_base_format,
                             gl_base_type,
                             in_data);

    gl_assert(glapi, leaving texture_image::clear_image_sub_data());
    return true;

#else // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
    return false;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_440
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
