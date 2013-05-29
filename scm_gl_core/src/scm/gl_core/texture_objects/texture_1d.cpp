
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_1d.h"

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

texture_1d_desc::texture_1d_desc(const unsigned     in_size,
                                 const data_format  in_format,
                                 const unsigned     in_mip_levels,
                                 const unsigned     in_array_layers)
  : _size(in_size)
  , _format(in_format)
  , _mip_levels(in_mip_levels)
  , _array_layers(in_array_layers)
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
  : texture_image(in_device)
  , _descriptor(in_desc)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    if (state().ok()) {
        allocate_storage(in_device, in_desc);
    }


    gl_assert(glapi, leaving texture_1d::texture_1d());
}

texture_1d::texture_1d(render_device&            in_device,
                       const texture_1d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
  : texture_image(in_device)
  , _descriptor(in_desc)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    
    if (state().ok()) {
        allocate_storage(in_device, in_desc);
    }
    if (state().ok()) {
        if (in_initial_mip_level_data.size() > 0) {
            upload_initial_data(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data);
        }
    }
    gl_assert(glapi, leaving texture_1d::texture_1d());
}

texture_1d::texture_1d(render_device&            in_device,
                       const texture_1d&         in_orig_texture,
                       const data_format         in_data_format,
                       const math::vec2ui&       in_mip_range,
                       const math::vec2ui&       in_layer_range)
  : texture_image(in_device)
  , _descriptor(in_orig_texture.descriptor())
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_430) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
    }
    else {
        if (state().ok()) {
            if(in_orig_texture.state().ok()) {
                create_texture_view(in_device, in_orig_texture, in_data_format, in_mip_range, in_layer_range);
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
            }
        }
    }
    gl_assert(glapi, leaving texture_1d::texture_1d());
}

texture_1d::~texture_1d()
{
}

const texture_1d_desc&
texture_1d::descriptor() const
{
    return _descriptor;
}

bool
texture_1d::allocate_storage(const render_device&      in_device,
                             const texture_1d_desc&    in_desc)
{
    assert(state().ok());

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);
    
    unsigned init_mip_levels = in_desc._mip_levels;
    if (init_mip_levels == 0) {
        init_mip_levels = util::max_mip_levels(in_desc._size);
    }
    else if (init_mip_levels != 1) {
        // assert(init_mip_levels == util::max_mip_levels(in_desc._size));
    }

    unsigned gl_internal_format = util::gl_internal_format(in_desc._format);
    unsigned gl_base_format     = util::gl_base_format(in_desc._format);
    unsigned gl_base_type       = util::gl_base_type(in_desc._format);

    if (in_desc._array_layers == 1) {
        context_bindable_object::_gl_object_target  = GL_TEXTURE_1D;
        context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_1D;

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
        util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
        glapi.glBindTexture(object_target(), object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS

        if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) { // false) { //BUG r280 
            //glerr() << "storage" << log::end;
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glTextureStorage1DEXT(object_id(), object_target(), init_mip_levels, gl_internal_format, in_desc._size);
            }
            else { // no EXT_DIRECT_STATE_ACCESS
                glapi.glTexStorage1D(object_target(), init_mip_levels, gl_internal_format, in_desc._size);
            }
            gl_assert(glapi, texture_1d::image_data() after glTexStorage1D());
        }
        else { // SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_420
            // make sure the unpack buffer is not bound!
            util::buffer_binding_guard upbg(glapi, GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING);
            glapi.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            for (unsigned i = 0; i < init_mip_levels; ++i) {
                unsigned lev_size = util::mip_level_dimensions(in_desc._size, i);
                if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                    glapi.glTextureImage1DEXT(object_id(), object_target(), i, gl_internal_format, lev_size, 0, gl_base_format, gl_base_type, 0);
                }
                else { 
                    glapi.glTexImage1D(object_target(), i, gl_internal_format, lev_size, 0, gl_base_format, gl_base_type, 0);
                }

                gl_assert(glapi, texture_1d::image_data() after glTexImage2D());
            }
        }
    }
    else if (in_desc._array_layers > 1) { // array textures
        context_bindable_object::_gl_object_target  = GL_TEXTURE_1D_ARRAY;
        context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_1D_ARRAY;

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
        util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
        glapi.glBindTexture(object_target(), object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
        if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {//false) { // BUG r280 
            //glerr() << "storage" << log::end;
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glTextureStorage2DEXT(object_id(), object_target(), init_mip_levels, gl_internal_format, in_desc._size, in_desc._array_layers);
            }
            else { // no EXT_DIRECT_STATE_ACCESS
                glapi.glTexStorage2D(object_target(), init_mip_levels, gl_internal_format, in_desc._size, in_desc._array_layers);
            }
            gl_assert(glapi, texture_1d::image_data() after glTexStorage2D());
        }
        else { // SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_420
            for (unsigned i = 0; i < init_mip_levels; ++i) {
                unsigned lev_size = util::mip_level_dimensions(in_desc._size, i);
                if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                    glapi.glTextureImage2DEXT(object_id(), object_target(), i, gl_internal_format, lev_size, in_desc._array_layers, 0, gl_base_format, gl_base_type, 0);
                }
                else {
                    glapi.glTexImage2D(object_target(), i, gl_internal_format, lev_size, in_desc._array_layers, 0, gl_base_format, gl_base_type, 0);
                }
                gl_assert(glapi, texture_1d::image_data() after glTexImage2D());
            }
        }
    }
    else {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }


    if (glerror) {
        state().set(glerror.to_object_state());
        return false;
    }
    else {
        return true;
    }
}

bool
texture_1d::upload_initial_data(const render_device&      in_device,
                                const texture_1d_desc&    in_desc,
                                const data_format         in_initial_data_format,
                                const std::vector<void*>& in_initial_mip_level_data)
{
    using namespace scm::math;

    for (int l = 0, s = static_cast<int>(in_initial_mip_level_data.size()); l < s; ++l) {
        const unsigned int   ls = util::mip_level_dimensions(in_desc._size, l);
        const texture_region r(vec3ui(0u), vec3ui(ls, in_desc._array_layers, 1));
        const void*          d = in_initial_mip_level_data[l];
        if (!image_sub_data(*(in_device.main_context()), r, l, in_initial_data_format, d)) {
            return false;
        }
    }

    return true;
}

bool
texture_1d::image_sub_data(const render_context& in_context,
                           const texture_region& in_region,
                           const unsigned        in_level,
                           const data_format     in_data_format,
                           const void*const      in_data)
{
    assert(state().ok());

    const opengl::gl_core& glapi = in_context.opengl_api();
    util::gl_error         glerror(glapi);
    
    unsigned gl_base_format = util::gl_base_format(in_data_format);
    unsigned gl_base_type   = util::gl_base_type(in_data_format);

    if (is_compressed_format(in_data_format)) {
        glerr() << log::error
                << "texture_1d::image_sub_data(): currently not supporting incoming compressed data formats" << log::end;
        return false;
    }

    if (_descriptor._array_layers == 1) {
        if (   in_region._origin.y != 0
            && in_region._origin.z != 0) {
            assert(in_region._origin.y == 0);
            assert(in_region._origin.z == 0);
            return false;
        }
        if (   in_region._dimensions.y != 1
            && in_region._dimensions.z != 1) {
            assert(in_region._dimensions.y == 1);
            assert(in_region._dimensions.z == 1);
            return false;
        }
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureSubImage1DEXT(object_id(), object_target(),
                                         in_level,
                                         in_region._origin.x,
                                         in_region._dimensions.x,
                                         gl_base_format,
                                         gl_base_type,
                                         in_data);
        }
        else {
            util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
            glapi.glBindTexture(object_target(), object_id());

            glapi.glTexSubImage1D(object_target(),
                                  in_level,
                                  in_region._origin.x,
                                  in_region._dimensions.x,
                                  gl_base_format,
                                  gl_base_type,
                                  in_data);
        }
        gl_assert(glapi, texture_1d::image_sub_data() after glTexSubImage1D());
    }
    else if (_descriptor._array_layers > 1) {
        if (in_region._origin.z != 0) {
            assert(in_region._origin.z == 0);
            return false;
        }
        if (in_region._dimensions.z != 1) {
            assert(in_region._dimensions.z == 1);
            return false;
        }
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureSubImage2DEXT(object_id(), object_target(),
                                         in_level,
                                         in_region._origin.x,     in_region._origin.y,
                                         in_region._dimensions.x, in_region._dimensions.y,
                                         gl_base_format,
                                         gl_base_type,
                                         in_data);
        }
        else {
            util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
            glapi.glBindTexture(object_target(), object_id());

            glapi.glTexSubImage2D(object_target(),
                                  in_level,
                                  in_region._origin.x,     in_region._origin.y,
                                  in_region._dimensions.x, in_region._dimensions.y,
                                  gl_base_format,
                                  gl_base_type,
                                  in_data);
        }
        gl_assert(glapi, texture_1d::image_sub_data() after glTexSubImage2D());
    }

    return true;
}

bool
texture_1d::create_texture_view(const render_device&      in_device,
                                const texture_1d&         in_orig_texture,
                                const data_format         in_data_format,
                                const math::vec2ui&       in_mip_range,
                                const math::vec2ui&       in_layer_range)
{
    assert(state().ok());

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error         glerror(glapi);
    
    if (in_mip_range.y <= in_mip_range.x) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }
    if (in_layer_range.y <= in_layer_range.x) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    const unsigned nb_mip_levels      =   math::min(_descriptor._mip_levels, in_mip_range.y)
                                        - math::min(_descriptor._mip_levels, in_mip_range.x);
    const unsigned nb_array_layers    =   math::min(_descriptor._array_layers, in_layer_range.y)
                                        - math::min(_descriptor._array_layers, in_layer_range.x);

    const unsigned gl_internal_format = util::gl_internal_format(in_data_format);

    if (nb_array_layers == 1) {
        context_bindable_object::_gl_object_target  = GL_TEXTURE_1D;
        context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_1D;
    }
    else if (nb_array_layers > 1) { // array texture
        context_bindable_object::_gl_object_target  = GL_TEXTURE_1D_ARRAY;
        context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_1D_ARRAY;
    }
    else {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    glapi.glTextureView(object_id(),
                        object_target(),
                        in_orig_texture.object_id(),
                        gl_internal_format,
                        in_mip_range.x,
                        nb_mip_levels,
                        in_layer_range.x,
                        nb_array_layers);

    gl_assert(glapi, texture_1d::create_texture_view() after glTextureView());

    if (glerror) {
        state().set(glerror.to_object_state());

        return false;
    }
    else {
        // setup view descriptor
        _descriptor._format       = in_data_format;
        _descriptor._mip_levels   = nb_mip_levels;
        _descriptor._array_layers = nb_array_layers;

        return true;
    }
}

data_format
texture_1d::format() const
{
    return _descriptor._format;
}

math::vec2ui
texture_1d::dimensions() const
{
    return math::vec2ui(_descriptor._size, 1);
}

unsigned
texture_1d::array_layers() const
{
    return _descriptor._array_layers;
}

unsigned
texture_1d::mip_map_layers() const
{
    return _descriptor._mip_levels;
}

unsigned
texture_1d::samples() const
{
    return 1;
}

} // namespace gl
} // namespace scm
