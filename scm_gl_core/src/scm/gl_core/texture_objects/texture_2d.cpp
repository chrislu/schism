
#include "texture_2d.h"

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>

namespace scm {
namespace gl {

texture_2d_desc::texture_2d_desc(const math::vec2ui& in_size,
                    const data_format   in_format,
                    const unsigned      in_mip_levels,
                    const unsigned      in_array_layers,
                    const unsigned      in_samples)
  : _size(in_size),
    _format(in_format),
    _mip_levels(in_mip_levels),
    _array_layers(in_array_layers),
    _samples(in_samples)
{
}

bool
texture_2d_desc::operator==(const texture_2d_desc& rhs) const
{
    return (   (_size         == rhs._size)
            && (_format       == rhs._format)
            && (_mip_levels   == rhs._mip_levels)
            && (_array_layers == rhs._array_layers)
            && (_samples      == rhs._samples));
}

bool
texture_2d_desc::operator!=(const texture_2d_desc& rhs) const
{
    return (   (_size         != rhs._size)
            || (_format       != rhs._format)
            || (_mip_levels   != rhs._mip_levels)
            || (_array_layers != rhs._array_layers)
            || (_samples      != rhs._samples));
}

texture_2d::texture_2d(render_device&           in_device,
                       const texture_2d_desc&   in_desc)
  : texture(in_device),
    _descriptor(in_desc)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();

    if (state().ok()) {
        image_data(in_device, in_desc, FORMAT_NULL, std::vector<void*>());
    }

    gl_assert(glapi, leaving texture_2d::texture_2d());
}

texture_2d::texture_2d(render_device&            in_device,
                       const texture_2d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
  : texture(in_device),
    _descriptor(in_desc)
{
    const opengl::gl3_core& glapi = in_device.opengl3_api();
    
    if (state().ok()) {
        image_data(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data);
    }

    gl_assert(glapi, leaving texture_2d::texture_2d());
}

texture_2d::~texture_2d()
{
}

const texture_2d_desc&
texture_2d::descriptor() const
{
    return (_descriptor);
}

bool
texture_2d::image_data(const render_device&      in_device,
                       const texture_2d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
{
    assert(state().ok());

    const opengl::gl3_core& glapi = in_device.opengl3_api();
    util::gl_error          glerror(glapi);

    if (in_desc._samples > static_cast<unsigned>(in_device.capabilities()._max_samples)) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (false);
    }

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
        if (in_desc._samples == 1) {

            _gl_object_target   = GL_TEXTURE_2D;
            _gl_texture_binding = GL_TEXTURE_BINDING_2D;

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
            glapi.glBindTexture(object_target(), object_id());
            
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            for (unsigned i = 0; i < init_mip_levels; ++i) {
                math::vec2ui lev_size = in_desc._size;
                if (i > 0) {
                    lev_size = util::mip_level_dimensions(in_desc._size, i);
                }
                const void* init_lev_data = inital_data ? in_initial_mip_level_data[i] : 0;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                glapi.glTextureImage2DEXT(object_id(), object_target(),
                                          i,
                                          util::gl_internal_format(in_desc._format),
                                          lev_size.x, lev_size.y,
                                          0,
                                          gl_base_format,
                                          gl_base_type,
                                          init_lev_data);
#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                glapi.glTexImage2D(object_target(),
                                   i,
                                   util::gl_internal_format(in_desc._format),
                                   lev_size.x, lev_size.y,
                                   0,
                                   gl_base_format,
                                   gl_base_type,
                                   init_lev_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }

                gl_assert(glapi, texture_2d::image_data() after glTexImage2D());
            }
        }
        // multi sample texture
        else if (in_desc._samples > 1) {
            if (in_desc._samples > in_device.capabilities()._max_samples) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                return (false);
            }
            if (in_desc._mip_levels == 1) {

                _gl_object_target   = GL_TEXTURE_2D_MULTISAMPLE;
                _gl_texture_binding = GL_TEXTURE_BINDING_2D_MULTISAMPLE;

                util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
                glapi.glBindTexture(object_target(), object_id());
                glapi.glTexImage2DMultisample(object_target(),
                                              in_desc._samples,
                                              util::gl_internal_format(in_desc._format),
                                              in_desc._size.x, in_desc._size.y, GL_FALSE);

                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }

                gl_assert(glapi, texture_2d::image_data() after glTexImage2DMultisample());
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                return (false);
            }
        }
        else {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }
    }
    // array textures
    else if (in_desc._array_layers > 1) {
        // non multi sample texture
        if (in_desc._samples == 1) {

            _gl_object_target   = GL_TEXTURE_2D_ARRAY;
            _gl_texture_binding = GL_TEXTURE_BINDING_2D_ARRAY;

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
            glapi.glBindTexture(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            for (unsigned i = 0; i < init_mip_levels; ++i) {
                math::vec2ui lev_size = in_desc._size;
                if (i > 0) {
                    lev_size = util::mip_level_dimensions(in_desc._size, i);
                }
                const void* init_lev_data = inital_data ? in_initial_mip_level_data[i] : 0;
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                glapi.glTextureImage3DEXT(object_id(), object_target(),
                                          i,
                                          util::gl_internal_format(in_desc._format),
                                          lev_size.x, lev_size.y, in_desc._array_layers,
                                          0,
                                          gl_base_format,
                                          gl_base_type,
                                          init_lev_data);

#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
                glapi.glTexImage3D(object_target(),
                                   i,
                                   util::gl_internal_format(in_desc._format),
                                   lev_size.x, lev_size.y, in_desc._array_layers,
                                   0,
                                   gl_base_format,
                                   gl_base_type,
                                   init_lev_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }

                gl_assert(glapi, texture_2d::image_data() after glTexImage3D());
            }
        }
        else if (in_desc._samples > 1) {
            if (in_desc._samples > in_device.capabilities()._max_samples) {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                return (false);
            }
            if (in_desc._mip_levels == 1) {

                _gl_object_target   = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
                _gl_texture_binding = GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;

                util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
                glapi.glBindTexture(object_target(), object_id());
                glapi.glTexImage3DMultisample(object_target(),
                                              in_desc._samples,
                                              util::gl_internal_format(in_desc._format),
                                              in_desc._size.x, in_desc._size.y, in_desc._array_layers, GL_FALSE);

                if (glerror) {
                    state().set(glerror.to_object_state());
                    return (false);
                }

                gl_assert(glapi, texture_2d::image_data() after glTexImage3DMultisample());
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
                return (false);
            }
        }
        else {
            state().set(object_state::OS_ERROR_INVALID_VALUE);
            return (false);
        }
    }
    else {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return (false);
    }

    gl_assert(glapi, leaving texture_2d::image_data());

    return (true);
}

bool
texture_2d::image_sub_data(const render_context& in_context,
                           const texture_region& in_region,
                           const unsigned        in_level,
                           const data_format     in_data_format,
                           const void*const      in_data)
{
    assert(state().ok());

    const opengl::gl3_core& glapi = in_context.opengl_api();
    util::gl_error          glerror(glapi);
    
    //unsigned gl_base_format = util::gl_base_format(_descriptor._format);
    //unsigned gl_base_type   = util::gl_base_type(_descriptor._format);

    unsigned gl_base_format = util::gl_base_format(in_data_format);
    unsigned gl_base_type   = util::gl_base_type(in_data_format);

    if (_descriptor._array_layers == 1) {
        if (in_region._origin.z != 0) {
            assert(in_region._origin.z == 0);
            return (false);
        }
        if (in_region._dimensions.z != 1) {
            assert(in_region._dimensions.z == 1);
            return (false);
        }

        if (_descriptor._samples == 1) {
#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
            glapi.glBindTexture(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTextureSubImage2DEXT(object_id(), object_target(),
                                         in_level,
                                         in_region._origin.x,     in_region._origin.y,
                                         in_region._dimensions.x, in_region._dimensions.y,
                                         gl_base_format,
                                         gl_base_type,
                                         in_data);
#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTexSubImage2D(object_target(),
                                  in_level,
                                  in_region._origin.x,     in_region._origin.y,
                                  in_region._dimensions.x, in_region._dimensions.y,
                                  gl_base_format,
                                  gl_base_type,
                                  in_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            gl_assert(glapi, texture_2d::image_sub_data() after glTexSubImage2D());
        }
        else {
            return (false);
        }
    }
    else if (_descriptor._array_layers > 1) {
        if (_descriptor._samples == 1) {

#ifndef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            util::texture_binding_guard save_guard(glapi, object_target(), texture_binding());
            glapi.glBindTexture(object_target(), object_id());
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTextureSubImage3DEXT(object_id(), object_target(),
                                         in_level,
                                         in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                         in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                         gl_base_format,
                                         gl_base_type,
                                         in_data);
#else // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            glapi.glTexSubImage3D(object_target(),
                                  in_level,
                                  in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                  in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                  gl_base_format,
                                  gl_base_type,
                                  in_data);
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
            gl_assert(glapi, texture_2d::image_sub_data() after glTexSubImage3D());
        }
        else {
            return (false);
        }
    }

    return (true);
}

data_format
texture_2d::format() const
{
    return (_descriptor._format);
}

math::vec2ui
texture_2d::dimensions() const
{
    return (_descriptor._size);
}

unsigned
texture_2d::array_layers() const
{
    return (_descriptor._array_layers);
}

unsigned
texture_2d::mip_map_layers() const
{
    return (_descriptor._mip_levels);
}

unsigned
texture_2d::samples() const
{
    return (_descriptor._samples);
}

} // namespace gl
} // namespace scm
