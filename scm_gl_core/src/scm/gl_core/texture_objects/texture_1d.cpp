
#include "texture_1d.h"

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>

namespace scm {
namespace gl {

texture_1d_desc::texture_1d_desc(const unsigned     in_size,
                                 const data_format  in_format,
                                 const unsigned     in_mip_levels,
                                 const unsigned     in_array_layers)
  : _size(in_size),
    _format(in_format),
    _mip_levels(in_mip_levels),
    _array_layers(in_array_layers)
{
}

bool
texture_1d_desc::operator==(const texture_1d_desc& rhs) const
{
    return (   (_size         == rhs._size)
            && (_format       == rhs._format)
            && (_mip_levels   == rhs._mip_levels)
            && (_array_layers == rhs._array_layers));
}

bool
texture_1d_desc::operator!=(const texture_1d_desc& rhs) const
{
    return (   (_size         != rhs._size)
            || (_format       != rhs._format)
            || (_mip_levels   != rhs._mip_levels)
            || (_array_layers != rhs._array_layers));
}

texture_1d::texture_1d(render_device&           in_device,
                       const texture_1d_desc&   in_desc)
  : texture(in_device),
    _descriptor(in_desc)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();

    if (state().ok()) {
        image_data(in_device, in_desc, FORMAT_NULL, std::vector<void*>());
    }

    gl_assert(glapi, leaving texture_1d::texture_1d());
}

texture_1d::texture_1d(render_device&            in_device,
                       const texture_1d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
  : texture(in_device),
    _descriptor(in_desc)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();
    
    if (state().ok()) {
        image_data(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data);
    }

    gl_assert(glapi, leaving texture_1d::texture_1d());
}

texture_1d::~texture_1d()
{
}

const texture_1d_desc&
texture_1d::descriptor() const
{
    return (_descriptor);
}

bool
texture_1d::image_data(const render_device&      in_device,
                       const texture_1d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
{
    assert(state().ok());

    const opengl::gl3_core& glapi = in_device.opengl3_api();
    util::gl_error          glerror(glapi);
    
    unsigned gl_base_format = 0;
    unsigned gl_base_type   = 0;
    if (FORMAT_NULL != in_initial_data_format) {
        gl_base_format = util::gl_base_format(in_initial_data_format);
        gl_base_type   = util::gl_base_type(in_initial_data_format);
    }
    else {
        gl_base_format = util::gl_base_format(in_desc._format);
        gl_base_type   = util::gl_base_type(in_desc._format);
    }

    unsigned init_mip_levels = in_desc._mip_levels;
    if (init_mip_levels == 0) {
        init_mip_levels = util::max_mip_levels(in_desc._size);
    }
    else if (init_mip_levels != 1) {
        // assert(init_mip_levels == util::max_mip_levels(in_desc._size));
    }

    bool inital_data = false;
    if (0 != in_initial_mip_level_data.size()) {
        if (in_initial_mip_level_data.size() == init_mip_levels) {
            inital_data = true;
        }
        else {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }
    }

    if (in_desc._array_layers == 1) {
        _gl_object_target   = GL_TEXTURE_1D;
        _gl_texture_binding = GL_TEXTURE_BINDING_1D;

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        util::texture_binding_guard(glapi, object_target(), texture_binding());
        glapi.glBindTexture(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

        for (unsigned i = 0; i < init_mip_levels; ++i) {
            unsigned lev_size = in_desc._size;
            if (i > 0) {
                lev_size = util::mip_level_dimensions(in_desc._size, i);
            }
            const void* init_lev_data = inital_data ? in_initial_mip_level_data[i] : 0;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTextureImage1DEXT(object_id(), object_target(),
                                      i,
                                      util::gl_internal_format(in_desc._format),
                                      lev_size,
                                      0,
                                      gl_base_format,
                                      gl_base_type,
                                      init_lev_data);
#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTexImage1D(object_target(),
                               i,
                               util::gl_internal_format(in_desc._format),
                               lev_size,
                               0,
                               gl_base_format,
                               gl_base_type,
                               init_lev_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            if (glerror) {
                state().set(glerror.to_object_state());
                return (false);
            }

            gl_assert(glapi, texture_1d::image_data() after glTexImage2D());
        }
    }
    // array textures
    else if (in_desc._array_layers > 1) {
        // non multi sample texture
        _gl_object_target   = GL_TEXTURE_1D_ARRAY;
        _gl_texture_binding = GL_TEXTURE_BINDING_1D_ARRAY;

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
        util::texture_binding_guard(glapi, object_target(), texture_binding());
        glapi.glBindTexture(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

        for (unsigned i = 0; i < init_mip_levels; ++i) {
            unsigned lev_size = in_desc._size;
            if (i > 0) {
                lev_size = util::mip_level_dimensions(in_desc._size, i);
            }
            const void* init_lev_data = inital_data ? in_initial_mip_level_data[i] : 0;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTextureImage2DEXT(object_id(), object_target(),
                                      i,
                                      util::gl_internal_format(in_desc._format),
                                      lev_size, in_desc._array_layers,
                                      0,
                                      gl_base_format,
                                      gl_base_type,
                                      init_lev_data);

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTexImage2D(object_target(),
                               i,
                               util::gl_internal_format(in_desc._format),
                               lev_size, in_desc._array_layers,
                               0,
                               gl_base_format,
                               gl_base_type,
                               init_lev_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            if (glerror) {
                state().set(glerror.to_object_state());
                return (false);
            }

            gl_assert(glapi, texture_1d::image_data() after glTexImage3D());
        }
    }
    else {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (false);
    }

    gl_assert(glapi, leaving texture_1d::image_data());

    return (true);
}

bool
texture_1d::image_sub_data(const render_context& in_context,
                           const texture_region& in_region,
                           const unsigned        in_level,
                           const void*const      in_data)
{
    return (false);
}

data_format
texture_1d::format() const
{
    return (_descriptor._format);
}

math::vec2ui
texture_1d::dimensions() const
{
    return (math::vec2ui(_descriptor._size, 1));
}

unsigned
texture_1d::array_layers() const
{
    return (_descriptor._array_layers);
}

unsigned
texture_1d::mip_map_layers() const
{
    return (_descriptor._mip_levels);
}

unsigned
texture_1d::samples() const
{
    return (1);
}

} // namespace gl
} // namespace scm